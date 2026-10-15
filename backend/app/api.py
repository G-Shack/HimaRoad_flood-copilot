"""One-snapshot bearer sessions for the trusted companion API."""

import hashlib
import os
import secrets
import sqlite3
from dataclasses import replace
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import Field, model_validator
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.app.copilot.chat import answer_question
from backend.app.copilot.offline import (
    NATIVE_REVIEW,
    Language,
    OfflineMenu,
    QuestionID,
    answer_predefined,
    offline_menu,
)
from backend.app.copilot.refusals import Refusal, refusal
from backend.app.copilot.rendering import prepare_turn, render_template
from backend.app.copilot.serving import ServingClient, ToolProposal
from backend.app.copilot.tools import (
    EvidenceContext,
    ToolRejected,
    dispatch,
    open_immutable_context,
)
from backend.app.reports.exports import gzip_chunks, iter_csv, iter_json
from backend.app.reports.facts import build_report
from backend.app.reports.pdf import ReportLayoutError, pdf_bytes
from backend.app.schemas import SHA256, Answer, Contract, EvidenceClaim, HimaRoadRun, ReportFacts
from backend.app.settings import Settings
from backend.app.storage import canonical_run, import_evidence, load_run_record, output_root

MAX_BODY_BYTES = 32 * 1024
MAX_RUN_BYTES = 1024**3
MAX_FILE_BYTES = 256 * 1024**2
MAX_JSON_BYTES = 16 * 1024**2
MAX_MEMBERS = 10_000
MAX_SESSIONS = 128
SESSION_SECONDS = 3600


class LoadRequest(Contract):
    path: str = Field(min_length=1, max_length=4096)


class LoadedRun(Contract):
    run: HimaRoadRun
    evidence_id: SHA256
    session_token: str
    expires_in_seconds: Literal[3600] = SESSION_SECONDS


class AskRequest(Contract):
    language: Language = "en"
    question_id: QuestionID | None = None
    question: str | None = Field(default=None, min_length=1, max_length=8000)
    selected_subject: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def one_question(self) -> "AskRequest":
        if (self.question_id is None) == (self.question is None):
            raise ValueError("Supply one predefined question or free-text question")
        return self


class RouteRequest(Contract):
    selected_subject: str = Field(min_length=1, max_length=512)
    scenario: Literal["baseline", "1", "2", "3", "4"] = "2"
    mode: Literal["vehicle", "pedestrian"] = "vehicle"
    language: Language = "en"


class AskResponse(Contract):
    evidence_id: SHA256
    answer: Answer | Refusal
    native_review: str = NATIVE_REVIEW
    predefined_menu: OfflineMenu


class BodyLimit:
    """Reject oversized streamed bodies before JSON parsing, without trusting headers."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        chunks = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return
            total += len(message.get("body", b""))
            if total > MAX_BODY_BYTES:
                await JSONResponse({"detail": "request_too_large"}, status_code=413)(
                    scope, receive, send
                )
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        consumed = False

        async def replay():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def preflight(path: Path, config: Settings) -> Path:
    from backend.app.upstream import budget

    # Preserve the published API budgets and test overrides.
    return budget.preflight(path, config, json_limit=MAX_JSON_BYTES)


def create_app(settings: Settings | None = None, *, client: ServingClient | None = None) -> FastAPI:
    config = settings or Settings()  # pyright: ignore[reportCallIssue]
    model = client or ServingClient(config.llm_base_url, config.llm_model)
    app = FastAPI(title="Flood copilot", description="Educational prototype; trusted companion")
    app.add_middleware(BodyLimit)
    # Only token digests are retained; requests never provide an authorization allowlist.
    sessions: dict[str, tuple[str, float]] = {}
    lock = Lock()
    bearer = HTTPBearer(auto_error=False)
    cached: dict[str, tuple[tuple, EvidenceContext, ReportFacts]] = {}

    def database_stamp() -> tuple:
        database = output_root(config) / "copilot.sqlite3"
        if database.is_symlink() or not database.is_file():
            raise ValueError("evidence database unavailable")
        return tuple(
            (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            if (info := p.stat() if p.exists() else None) is not None
            else None
            for p in (database, Path(str(database) + "-wal"))
        )

    def snapshot(identifier: str) -> tuple[EvidenceContext, ReportFacts]:
        stamp = database_stamp()
        with lock:
            hit = cached.get(identifier)
            if hit is not None and hit[0] == stamp:
                return hit[1], hit[2]
        ctx = open_immutable_context(identifier, frozenset({identifier}), config)
        report = build_report(ctx, frozenset({identifier}))
        if database_stamp() != stamp:
            raise ValueError("evidence changed while verifying")
        with lock:
            if len(cached) >= 2:
                cached.clear()
            cached[identifier] = (stamp, ctx, report)
        return ctx, report

    def subject_context(ctx: EvidenceContext, subject: str | None) -> EvidenceContext:
        if subject is not None and not any(
            c.kind == "feature" and c.subject == subject for c in ctx.claims
        ):
            raise ToolRejected("feature_outside_run")
        return replace(ctx, selected_subject=subject)

    @app.middleware("http")
    async def no_store(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        # FastAPI's default response includes the supplied input (possibly secrets).
        return JSONResponse({"detail": "invalid_request"}, status_code=422)

    def context(
        credential: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> EvidenceContext:
        if credential is None or len(credential.credentials) > 128:
            raise HTTPException(401, "session_required")
        digest = hashlib.sha256(credential.credentials.encode()).hexdigest()
        with lock:
            saved = sessions.get(digest)
            if saved is None or saved[1] <= monotonic():
                sessions.pop(digest, None)
                raise HTTPException(401, "session_expired_or_unknown")
        identifier = saved[0]
        try:
            return snapshot(identifier)[0]
        except (ValueError, sqlite3.Error, OSError):
            raise HTTPException(409, "evidence_unavailable") from None

    def facts(ctx: EvidenceContext) -> ReportFacts:
        return snapshot(ctx.evidence_id)[1]

    def permitted_claims(ctx: EvidenceContext) -> tuple[EvidenceClaim, ...]:
        selected = facts(ctx).claims
        incomplete = any(c.kind == "status" and c.value == "incomplete" for c in ctx.claims)
        failed = any(
            c.kind == "evidence_validation" and c.value["result"] == "failed" for c in ctx.claims
        )
        if incomplete and not failed:
            # Unknown-only derived bands are useful in answers, while reports omit them.
            from backend.app.analysis.sensitivity import ExposureBand

            for candidate in ctx.claims:
                if candidate.kind == "sensitivity":
                    band = ExposureBand.model_validate(candidate.value)
                    if all(
                        getattr(band, key) is None
                        for key in (
                            "water_only",
                            "water_or_disturbance",
                            "unobservable_features",
                            "feature_count",
                            "partial_coverage_features",
                        )
                    ):
                        selected += (candidate,)
                elif (
                    candidate.kind == "flow_path"
                    and candidate.value.get("availability") == "unavailable"
                ):
                    selected += (candidate,)
        return selected

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runs")
    def runs() -> dict:
        # Names only; never open an unknown run/reference to populate a picker.
        paths = []
        scanned = 0
        for folder, directories, files in os.walk(config.himaroad_data_dir, followlinks=False):
            directories[:] = sorted(
                name
                for name in directories
                if name.casefold() != "reference" and not (Path(folder) / name).is_symlink()
            )
            scanned += 1
            if scanned > MAX_MEMBERS:
                return {"paths": paths, "truncated": True}
            if {"report.json", "manifest.json"} <= set(files):
                relative = Path(folder).relative_to(config.himaroad_data_dir).as_posix()
                paths.append(relative)
                directories.clear()
        return {"paths": sorted(paths), "truncated": False}

    @app.post("/api/runs/load", response_model=LoadedRun)
    def load(request: LoadRequest) -> LoadedRun:
        with lock:
            for key in [key for key, saved in sessions.items() if saved[1] <= monotonic()]:
                del sessions[key]
            if len(sessions) >= MAX_SESSIONS:
                raise HTTPException(429, "session_limit")
        try:
            from backend.app.preparation import prepared_run

            saved = prepared_run(Path(request.path), config, verify_evidence=False)
            if saved is None:
                path = preflight(Path(request.path), config)
                identifier, claims = import_evidence(path, config)
                run = load_run_record(claims[0].run_id, config)
            else:
                identifier, run = saved
            # Complete verification/pre-warm occurs before issuing a capability.
            ctx, _ = snapshot(identifier)
            if ctx.claims[0].run_id != hashlib.sha256(canonical_run(run)).hexdigest():
                raise ValueError("prepared metadata does not match evidence")
        except (ValueError, OSError, sqlite3.Error):
            raise HTTPException(422, "run_import_refused") from None
        token = secrets.token_urlsafe(32)
        with lock:
            if len(sessions) >= MAX_SESSIONS:
                raise HTTPException(429, "session_limit")
            sessions[hashlib.sha256(token.encode()).hexdigest()] = (
                identifier,
                monotonic() + SESSION_SECONDS,
            )
        return LoadedRun(run=run, evidence_id=identifier, session_token=token)

    @app.delete("/api/session", status_code=204)
    def close(
        ctx: Annotated[EvidenceContext, Depends(context)],
        credential: Annotated[HTTPAuthorizationCredentials, Depends(bearer)],
    ) -> Response:
        with lock:
            sessions.pop(hashlib.sha256(credential.credentials.encode()).hexdigest(), None)
        return Response(status_code=204)

    @app.get("/api/evidence", response_model=ReportFacts)
    def evidence(ctx: Annotated[EvidenceContext, Depends(context)]) -> ReportFacts:
        return facts(ctx)

    @app.get("/api/view")
    def view(ctx: Annotated[EvidenceContext, Depends(context)], selected: SHA256 | None = None):
        from backend.app.reports.view import _project_verified_view

        return _project_verified_view(facts(ctx), selected)

    @app.get("/api/claims/{claim_id}", response_model=EvidenceClaim)
    def claim(claim_id: SHA256, ctx: Annotated[EvidenceContext, Depends(context)]) -> EvidenceClaim:
        match = next((c for c in permitted_claims(ctx) if c.id == claim_id), None)
        if match is None:
            raise HTTPException(404, "claim_outside_session")
        return match

    @app.post("/api/ask", response_model=AskResponse)
    def ask(request: AskRequest, ctx: Annotated[EvidenceContext, Depends(context)]) -> AskResponse:
        ids = frozenset({ctx.evidence_id})
        try:
            selected = subject_context(ctx, request.selected_subject)
        except ToolRejected:
            raise HTTPException(422, "feature_outside_session") from None
        if request.question_id is not None:
            answer = render_template(
                prepare_turn(
                    answer_predefined(request.question_id, selected.claims), selected, ids
                ),
                request.language,
            )
        else:
            assert request.question is not None
            answer = answer_question(request.question, request.language, selected, ids, model)
        allowed = {c.id for c in permitted_claims(ctx)}
        if any(identifier not in allowed for identifier in answer.claim_ids):
            answer = refusal("evidence_unavailable", request.language)
        return AskResponse(
            evidence_id=ctx.evidence_id,
            answer=answer,
            predefined_menu=offline_menu(request.language),
        )

    @app.post("/api/route", response_model=AskResponse)
    def route(
        request: RouteRequest, ctx: Annotated[EvidenceContext, Depends(context)]
    ) -> AskResponse:
        ids = frozenset({ctx.evidence_id})
        try:
            selected = subject_context(ctx, request.selected_subject)
            result = dispatch(
                ToolProposal(
                    name="get_route",
                    arguments={
                        "evidence_id": ctx.evidence_id,
                        "feature": "selected_asset",
                        "scenario": request.scenario,
                        "mode": request.mode,
                    },
                ),
                selected,
                ids,
            )
            answer = render_template(prepare_turn(result, selected, ids), request.language)
        except ToolRejected:
            raise HTTPException(422, "route_scope_unavailable") from None
        if any(i not in {c.id for c in permitted_claims(ctx)} for i in answer.claim_ids):
            answer = refusal("evidence_unavailable", request.language)
        return AskResponse(
            evidence_id=ctx.evidence_id,
            answer=answer,
            predefined_menu=offline_menu(request.language),
        )

    @app.get("/api/report")
    def report(
        ctx: Annotated[EvidenceContext, Depends(context)],
        language: Language = "en",
        format: Literal["pdf", "json", "csv"] = "pdf",
        compression: Literal["none", "gzip"] = "none",
    ) -> Response:
        report_facts = facts(ctx)
        if format == "json":
            chunks, mime = iter_json(report_facts), "application/json"
        elif format == "csv":
            chunks, mime = iter_csv(report_facts), "text/csv"
        else:
            if compression != "none":
                raise HTTPException(422, "report_compression_unsupported")
            if config.report_font is None:
                raise HTTPException(503, "report_font_unavailable")
            try:
                content = pdf_bytes(report_facts, language, config.report_font)
            except (ReportLayoutError, ValueError, OSError):
                raise HTTPException(422, "report_render_refused") from None
            return Response(
                content,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="report-{language}.pdf"'},
            )
        suffix = format
        if compression == "gzip":
            chunks, mime, suffix = gzip_chunks(chunks), "application/gzip", format + ".gz"
        return StreamingResponse(
            chunks,
            media_type=mime,
            headers={"Content-Disposition": f'attachment; filename="report-{language}.{suffix}"'},
        )

    return app
