"""Cliente mínimo da API GraphQL do Linear.

Três detalhes que economizam depuração:

- A API é GraphQL: único endpoint, sempre POST — nem GET /issues.
- Chave pessoal vai crua no header Authorization, sem prefixo "Bearer".
  (Bearer só se aplica a tokens de OAuth.)
- Estados e labels são identificados por UUID, não por nome. O cliente resolve
  e mantém em cache o mapeamento.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import requests

DEFAULT_ENDPOINT = "https://api.linear.app/graphql"


class LinearError(RuntimeError):
    pass


class LinearClient:
    def __init__(
        self,
        team_key: str,
        api_key: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: int = 30,
    ) -> None:
        self.team_key = team_key
        self.api_key = api_key or os.environ.get("LINEAR_API_KEY")
        if not self.api_key:
            raise LinearError(
                "LINEAR_API_KEY não configurada. Exporte a variável de ambiente com a chave pessoal."
            )
        self.endpoint = endpoint
        self.timeout = timeout

    # --- HTTP ---------------------------------------------------------------

    def _post(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        resp = requests.post(
            self.endpoint,
            headers={"Authorization": self.api_key, "Content-Type": "application/json"},
            json={"query": query, "variables": variables or {}},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if "errors" in payload:
            raise LinearError(str(payload["errors"]))
        return payload["data"]

    # --- resolução de IDs ---------------------------------------------------

    @lru_cache(maxsize=1)
    def team_id(self) -> str:
        data = self._post(
            "query($key:String!){teams(filter:{key:{eq:$key}}){nodes{id key name}}}",
            {"key": self.team_key},
        )
        nodes = data["teams"]["nodes"]
        if not nodes:
            raise LinearError(f"time '{self.team_key}' não encontrado")
        return nodes[0]["id"]

    @lru_cache(maxsize=1)
    def states(self) -> dict[str, str]:
        """Mapa ``{nome_estado: uuid}`` do workflow do time.

        Consulta via ``team(id).states`` — o filtro em ``workflowStates`` no
        root da API mudou entre versões e passou a devolver 400.
        """
        data = self._post(
            "query($tid:String!){team(id:$tid){states{nodes{id name}}}}",
            {"tid": self.team_id()},
        )
        return {n["name"]: n["id"] for n in data["team"]["states"]["nodes"]}

    @lru_cache(maxsize=1)
    def labels(self) -> dict[str, str]:
        data = self._post(
            "query($tid:String!){team(id:$tid){labels{nodes{id name}}}}",
            {"tid": self.team_id()},
        )
        return {n["name"]: n["id"] for n in data["team"]["labels"]["nodes"]}

    # --- operações ----------------------------------------------------------

    def create_issue(
        self,
        title: str,
        description: str = "",
        state: str | None = None,
        labels: list[str] | None = None,
        parent_id: str | None = None,
    ) -> str:
        variables: dict[str, Any] = {
            "input": {
                "teamId": self.team_id(),
                "title": title,
                "description": description,
            }
        }
        if state is not None:
            variables["input"]["stateId"] = self.states()[state]
        if labels:
            label_map = self.labels()
            variables["input"]["labelIds"] = [label_map[n] for n in labels if n in label_map]
        if parent_id is not None:
            variables["input"]["parentId"] = parent_id
        data = self._post(
            "mutation($input:IssueCreateInput!){issueCreate(input:$input){success issue{id identifier}}}",
            variables,
        )
        issue = data["issueCreate"]["issue"]
        return issue["id"]

    def create_state(self, name: str, type_: str, color: str = "#95a5a6", position: float | None = None) -> str:
        """Cria um state no workflow do time. ``type_`` deve ser um dos:
        ``triage``, ``backlog``, ``unstarted``, ``started``, ``completed``, ``canceled``.
        """
        input_: dict[str, Any] = {
            "teamId": self.team_id(),
            "name": name,
            "type": type_,
            "color": color,
        }
        if position is not None:
            input_["position"] = position
        data = self._post(
            "mutation($input:WorkflowStateCreateInput!){workflowStateCreate(input:$input){success workflowState{id name}}}",
            {"input": input_},
        )
        state = data["workflowStateCreate"]["workflowState"]
        self.states.cache_clear()
        return state["id"]

    def create_label(self, name: str, color: str = "#95a5a6") -> str:
        """Cria uma label se ainda não existir; retorna o UUID."""
        existing = self.labels()
        if name in existing:
            return existing[name]
        data = self._post(
            "mutation($input:IssueLabelCreateInput!){issueLabelCreate(input:$input){success issueLabel{id name}}}",
            {"input": {"teamId": self.team_id(), "name": name, "color": color}},
        )
        label = data["issueLabelCreate"]["issueLabel"]
        self.labels.cache_clear()
        return label["id"]

    def move(self, issue_id: str, state: str) -> None:
        self._post(
            "mutation($id:String!,$sid:String!){issueUpdate(id:$id,input:{stateId:$sid}){success}}",
            {"id": issue_id, "sid": self.states()[state]},
        )

    def comment(self, issue_id: str, body: str) -> None:
        self._post(
            "mutation($input:CommentCreateInput!){commentCreate(input:$input){success}}",
            {"input": {"issueId": issue_id, "body": body}},
        )

    def count_issues_by_label(self, label: str) -> int:
        """Conta issues do time com uma label específica (todos os states)."""
        data = self._post(
            "query($tid:String!,$f:IssueFilter!){team(id:$tid){issues(filter:$f){nodes{id}}}}",
            {"tid": self.team_id(), "f": {"labels": {"name": {"eq": label}}}},
        )
        return len(data["team"]["issues"]["nodes"])

    def upload_file(self, path: "Path", content_type: str | None = None) -> str:
        """Sobe um arquivo para o Linear e devolve o assetUrl (URL pública p/ embed).

        Usa o fluxo de 3 passos do Linear:
          1. mutation fileUpload → recebe uploadUrl (S3 pré-assinado) + assetUrl
          2. PUT do binário para uploadUrl com os headers exigidos
          3. o assetUrl fica disponível para markdown ``![](assetUrl)``
        """
        import mimetypes
        from pathlib import Path

        p = Path(path)
        size = p.stat().st_size
        ctype = content_type or mimetypes.guess_type(p.name)[0] or "application/octet-stream"

        data = self._post(
            "mutation($c:String!,$f:String!,$s:Int!){fileUpload(contentType:$c,filename:$f,size:$s){success uploadFile{uploadUrl assetUrl headers{key value}}}}",
            {"c": ctype, "f": p.name, "s": size},
        )
        info = data["fileUpload"]["uploadFile"]
        headers = {h["key"]: h["value"] for h in info["headers"]}
        put_resp = requests.put(info["uploadUrl"], data=p.read_bytes(), headers=headers, timeout=self.timeout * 3)
        put_resp.raise_for_status()
        return info["assetUrl"]

    def issues_in_state(self, state: str, label: str | None = None) -> list[dict[str, Any]]:
        state_id = self.states()[state]
        filt: dict[str, Any] = {"state": {"id": {"eq": state_id}}}
        if label:
            filt["labels"] = {"name": {"eq": label}}
        data = self._post(
            "query($tid:String!,$f:IssueFilter!){team(id:$tid){issues(filter:$f){nodes{id identifier title}}}}",
            {"tid": self.team_id(), "f": filt},
        )
        return data["team"]["issues"]["nodes"]
