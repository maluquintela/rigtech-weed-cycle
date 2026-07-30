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
        """Mapa ``{nome_estado: uuid}`` do workflow do time."""
        data = self._post(
            "query($tid:String!){workflowStates(filter:{team:{id:{eq:$tid}}}){nodes{id name}}}",
            {"tid": self.team_id()},
        )
        return {n["name"]: n["id"] for n in data["workflowStates"]["nodes"]}

    @lru_cache(maxsize=1)
    def labels(self) -> dict[str, str]:
        data = self._post(
            "query($tid:String!){issueLabels(filter:{team:{id:{eq:$tid}}}){nodes{id name}}}",
            {"tid": self.team_id()},
        )
        return {n["name"]: n["id"] for n in data["issueLabels"]["nodes"]}

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

    def issues_in_state(self, state: str, label: str | None = None) -> list[dict[str, Any]]:
        state_id = self.states()[state]
        filt: dict[str, Any] = {"state": {"id": {"eq": state_id}}, "team": {"id": {"eq": self.team_id()}}}
        if label:
            filt["labels"] = {"name": {"eq": label}}
        data = self._post(
            "query($f:IssueFilter!){issues(filter:$f){nodes{id identifier title}}}",
            {"f": filt},
        )
        return data["issues"]["nodes"]
