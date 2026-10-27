"""LLM Router RL dashboard (Reflex).

Layout of this file
    1. DashboardState   all UI state + the one event handler that runs the router
    2. UI helpers       small reusable pieces (card, tiles, rows)
    3. Sections         one function per dashboard block
    4. Page + app       wiring

To connect your own RL agent / LLM calls, edit ONLY `backend.py`.
"""
import asyncio
from typing import Any

import reflex as rx

from . import backend

DEFAULT_PROMPT = "Derive the Bellman optimality equation and explain its use in DQN."


# ==========================================================================
# 1. STATE
# ==========================================================================
class DashboardState(rx.State):
    # ---- navigation / inputs ----
    active_nav: str = "Home"
    prompt: str = DEFAULT_PROMPT
    show_all: bool = True            # call every tier (for comparison) or only the selected one

    # ---- status ----
    is_loading: bool = False
    has_result: bool = False
    connected: bool = False
    summary: str = ""

    # ---- model pool (static description from backend.TIERS) ----
    models: list[dict[str, str]] = [
        {k: t[k] for k in ("tier", "color", "hex", "model_id", "params")} for t in backend.TIERS
    ]

    # ---- routing decision ----
    selected_tier: str = ""
    selected_color: str = "blue"
    n_actions: int = len(backend.TIERS)
    chart_data: list[dict[str, Any]] = []        # [{"tier": "SMALL (0)", "score": 0.0, "fill": "#60a5fa"}, ...]
    y_domain: list[float] = [-1.0, 1.0]
    score_rows: list[dict[str, str]] = []        # table under "Action Scores"
    reasons: list[str] = []                      # "Why This Model?" checklist
    analysis: list[dict[str, str]] = []          # "Prompt Analysis" rows

    # ---- model responses (one dict per tier) ----
    # keys: tier, color, hex, model_id, status(pending|skipped|ok|error), text, error, meta
    responses: list[dict[str, str]] = []

    # ------------------------------------------------------------------ events
    @rx.event
    def init_page(self):
        self.connected = backend.is_connected()

    @rx.event
    def set_prompt(self, value: str):
        self.prompt = value

    @rx.event
    def set_show_all(self, value: bool):
        self.show_all = value

    @rx.event
    def set_active_nav(self, name: str):
        self.active_nav = name

    @rx.event(background=True)
    async def run_router(self):
        """Prompt -> RL decision -> (one or all) LLM calls. Blocking work runs in threads."""
        async with self:
            prompt = self.prompt.strip()
            if not prompt or self.is_loading:
                return
            self.is_loading = True
            self.has_result = False
            self.summary = ""
            self.responses = [self._response_row(i, status="pending") for i in range(len(backend.TIERS))]

        try:
            # ---- 1) RL agent decision  (backend.route_prompt) ----
            decision = await asyncio.to_thread(backend.route_prompt, prompt)
            async with self:
                self._apply_decision(decision)

            # ---- 2) LLM calls         (backend.call_model) ----
            async with self:
                targets = list(range(len(backend.TIERS))) if self.show_all else [decision.action]
            results = await asyncio.gather(
                *[asyncio.to_thread(backend.call_model, i, prompt) for i in targets])
            async with self:
                rows = [self._response_row(i, status="skipped") for i in range(len(backend.TIERS))]
                for i, res in zip(targets, results):
                    rows[i] = self._response_row(i, res)
                self.responses = rows
        except Exception as exc:  # noqa: BLE001
            async with self:
                self.summary = f"Router error: {exc}"
        finally:
            async with self:
                self.is_loading = False

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _response_row(i: int, res: "backend.ModelResult | None" = None, status: str = "") -> dict[str, str]:
        t = backend.TIERS[i]
        row = {"tier": t["tier"], "color": t["color"], "hex": t["hex"], "model_id": t["model_id"],
               "status": status, "text": "", "error": "", "meta": ""}
        if res is not None:
            row["status"] = "error" if res.error else "ok"
            row["text"], row["error"] = res.text, res.error
            row["meta"] = f"{res.tokens} tokens  •  ${res.cost:.6f}"
        return row

    def _apply_decision(self, d: "backend.RoutingDecision"):
        a = d.action
        tier = backend.TIERS[a]
        self.selected_tier, self.selected_color = tier["tier"], tier["color"]
        self.chart_data = [
            {"tier": f"{t['tier']} ({i})", "score": round(s, 5), "fill": t["hex"]}
            for i, (t, s) in enumerate(zip(backend.TIERS, d.scores))]
        limit = max(1.0, max(abs(s) for s in d.scores) * 1.2)
        self.y_domain = [-round(limit, 2), round(limit, 2)]
        self.score_rows = [
            {"tier": t["tier"], "color": t["color"], "hex": t["hex"], "action": str(i),
             "score": f"{s:.5f}", "selected": "1" if i == a else "0"}
            for i, (t, s) in enumerate(zip(backend.TIERS, d.scores))]

        yes = [k for k, v in d.analysis.items() if v == "Yes"]
        self.reasons = [
            f"Highest learned score ({d.scores[a]:.5f}) among all actions",
            f"Good fit for the prompt's complexity ({' + '.join(k.lower() for k in yes) or 'general query'})",
            "Balanced performance for the detected task type",
            "Selected based on learned policy, not estimated cost or quality",
        ]
        self.analysis = [
            {"label": k, "value": v,
             "color": "green" if v == "Yes" else "gray" if v == "No" else "blue"}
            for k, v in d.analysis.items()]
        self.summary = (f"The RL agent analyzed the prompt features, compared the action scores for all "
                        f"available models, and routed your request to the {tier['tier']} model because "
                        f"it had the highest learned score.")
        self.has_result = True


S = DashboardState


# ==========================================================================
# 2. UI HELPERS
# ==========================================================================
def card(title: str, icon: str, *children, right=None, **props) -> rx.Component:
    """White metric card with a header row (icon + title + optional right-hand widget)."""
    header = [rx.icon(icon, size=18, color="var(--blue-10)"),
              rx.heading(title, size="4", color="var(--slate-12)"), rx.spacer()]
    if right is not None:
        header.append(right)
    return rx.card(
        rx.vstack(rx.hstack(*header, align="center", width="100%"), *children,
                  spacing="3", align="stretch", width="100%"),
        variant="surface", width="100%", background="white",
        border="1px solid var(--slate-5)", box_shadow="0 1px 2px rgba(15,23,42,.04)", **props)


def dot(color) -> rx.Component:
    return rx.box(width="10px", height="10px", border_radius="50%", background=color, flex_shrink="0")


def empty_hint(text: str) -> rx.Component:
    return rx.text(text, size="2", color="var(--slate-9)")


# ==========================================================================
# 3. SECTIONS
# ==========================================================================
def header() -> rx.Component:
    return rx.hstack(
        rx.icon("brain-circuit", size=34, color="#7dd3fc"),
        rx.vstack(rx.heading("LLM Router RL", size="5", color="white"),
                  rx.text("Smart Model Selection with Reinforcement Learning", size="2", color="#cbd5e1"),
                  spacing="0"),
        rx.spacer(),
        rx.hstack(
            rx.box(width="8px", height="8px", border_radius="50%",
                   background=rx.cond(S.connected, "#22c55e", "#f59e0b")),
            rx.text(rx.cond(S.connected, "Connected to OpenRouter", "Mock mode (no API key)"),
                    size="2", color="#e2e8f0"),
            align="center", spacing="2"),
        align="center", padding="14px 24px", width="100%", background="#1b2a49")


def nav_item(name: str, icon: str) -> rx.Component:
    active = S.active_nav == name
    return rx.hstack(
        rx.icon(icon, size=18), rx.text(name, size="3", weight="medium"),
        on_click=S.set_active_nav(name), align="center", spacing="3", cursor="pointer",
        padding="10px 14px", border_radius="10px", width="100%",
        color=rx.cond(active, "var(--blue-11)", "var(--slate-11)"),
        background=rx.cond(active, "var(--blue-3)", "transparent"),
        _hover={"background": "var(--slate-3)"})


def sidebar() -> rx.Component:
    return rx.vstack(
        nav_item("Home", "house"), nav_item("Model Pool", "boxes"),
        nav_item("Settings", "settings"), nav_item("About", "info"),
        spacing="1", padding="16px 12px", width="210px", min_width="210px",
        min_height="calc(100vh - 68px)", background="white", border_right="1px solid var(--slate-5)",
        display=rx.breakpoints(initial="none", md="flex"))


def prompt_card() -> rx.Component:
    return card(
        "Enter your prompt", "message-square-text",
        rx.text("Type any question or task below and see how the RL agent routes it to the right model.",
                size="2", color="var(--slate-10)"),
        rx.hstack(
            rx.text_area(value=S.prompt, on_change=S.set_prompt, rows="3", width="100%",
                         placeholder="Ask anything…"),
            rx.button(rx.icon("send", size=16), "Run", on_click=S.run_router, loading=S.is_loading,
                      size="3", style={"min_width": "100px"}),
            align="start", spacing="3", width="100%"))


def decision_card() -> rx.Component:
    chart = rx.recharts.bar_chart(
        rx.recharts.cartesian_grid(stroke_dasharray="3 3", vertical=False),
        rx.recharts.x_axis(data_key="tier"),
        rx.recharts.y_axis(domain=S.y_domain),
        rx.recharts.reference_line(y=0, stroke="#94a3b8"),
        rx.recharts.graphing_tooltip(),
        # each bar takes its colour from the "fill" key of its data row
        rx.recharts.bar(rx.recharts.label_list(data_key="score", position="top"),
                        data_key="score", radius=4),
        data=S.chart_data, width="100%", height=280)

    decision_box = rx.vstack(
        rx.hstack(rx.icon("trophy", size=18, color="var(--blue-10)"),
                  rx.heading("Decision", size="3"), align="center"),
        rx.cond(
            S.has_result,
            rx.text(rx.text.strong(S.selected_tier, color="var(--" + S.selected_color + "-11)"),
                    " was selected because its learned score is the highest among the ",
                    S.n_actions, " actions.", size="2", line_height="1.7"),
            empty_hint("Run the router to see the decision.")),
        padding="16px", border_radius="12px", background="var(--blue-2)",
        border="1px solid var(--blue-5)", align="start", width="100%")

    return card(
        "RL Agent Decision", "target",
        rx.text("Learned action scores ", rx.text.span("(not quality or cost predictions):",
                                                        color="var(--slate-9)"), size="2"),
        rx.cond(
            S.has_result,
            rx.grid(chart, decision_box, columns=rx.breakpoints(initial="1", md="2fr 1fr"),
                    spacing="4", width="100%", align="center"),
            rx.center(empty_hint("No decision yet — enter a prompt and press Run."), height="200px")))


# ---- Model responses ------------------------------------------------------
def error_card(r) -> rx.Component:
    return rx.box(
        rx.hstack(rx.icon("circle-alert", size=16, color="var(--red-10)"),
                  rx.text("API Error", weight="bold", size="2", color="var(--red-11)"), align="center"),
        rx.text(r["error"], size="1", color="var(--red-11)", word_break="break-word", margin_top="6px"),
        padding="12px", border_radius="8px", background="var(--red-3)", border="1px solid var(--red-6)",
        width="100%")


def response_card(r) -> rx.Component:
    body = rx.match(
        r["status"],
        ("error", error_card(r)),
        ("pending", rx.hstack(rx.spinner(size="2"), empty_hint("Waiting for response…"), align="center")),
        ("skipped", empty_hint("Not called. Turn on “Show all responses” to compare tiers.")),
        # rx.markdown renders **bold**, ``` code blocks ``` and $$ LaTeX $$ out of the box
        rx.box(rx.markdown(r["text"]), max_height="280px", overflow_y="auto", font_size="13px"))
    return rx.vstack(
        rx.hstack(dot(r["hex"]), rx.text(r["tier"], weight="bold", size="2", color="var(--slate-12)"),
                  rx.text("(", r["model_id"], ")", size="1", color="var(--slate-9)", word_break="break-all"),
                  align="center", spacing="2", wrap="wrap"),
        body, rx.spacer(),
        rx.cond(r["meta"] != "", rx.text(r["meta"], size="1", color="var(--slate-9)")),
        padding="14px", border_radius="12px", border="1px solid var(--slate-5)", background="var(--slate-1)",
        align="stretch", min_height="230px", width="100%", spacing="2")


def responses_card() -> rx.Component:
    toggle = rx.hstack(rx.icon("eye", size=16), rx.text("Show all responses", size="2"),
                       rx.switch(checked=S.show_all, on_change=S.set_show_all, size="1"),
                       align="center", spacing="2")
    return card(
        "Model Responses", "messages-square",
        rx.cond(
            S.responses.length() > 0,
            rx.grid(rx.foreach(S.responses, response_card),
                    columns=rx.breakpoints(initial="1", md="3"), spacing="3", width="100%"),
            empty_hint("Responses will appear here.")),
        right=toggle)


# ---- Right column --------------------------------------------------------
def model_tile(m) -> rx.Component:
    return rx.vstack(
        rx.badge(m["tier"], color_scheme=m["color"], variant="soft", size="2"),
        rx.text(m["model_id"], size="1", text_align="center", word_break="break-all"),
        rx.text("(", m["params"], ")", size="1", color="var(--slate-9)"),
        align="center", spacing="2", padding="12px", flex="1", border_radius="10px",
        border="1px solid var(--slate-5)", background="var(--slate-1)")


def pool_card() -> rx.Component:
    return card(
        "Model Pool", "chart-column",
        rx.hstack(rx.foreach(S.models, model_tile), spacing="3", width="100%", align="stretch"),
        right=rx.button("View models", variant="soft", size="1", on_click=S.set_active_nav("Model Pool")))


def score_row(row) -> rx.Component:
    return rx.table.row(
        rx.table.cell(rx.hstack(dot(row["hex"]), rx.text(row["tier"], size="2", weight="medium"),
                                align="center", spacing="2")),
        rx.table.cell(row["action"]),
        rx.table.cell(rx.code(row["score"], variant="ghost")),
        rx.table.cell(rx.cond(row["selected"] == "1",
                              rx.badge("SELECTED", color_scheme=row["color"], variant="soft"),
                              rx.text("–", color="var(--slate-8)"))),
        align="center")


def scores_card() -> rx.Component:
    return card(
        "Action Scores", "chart-no-axes-column",
        rx.cond(
            S.has_result,
            rx.table.root(
                rx.table.header(rx.table.row(
                    *[rx.table.column_header_cell(h) for h in ("Model Tier", "Action", "Score", "Status")])),
                rx.table.body(rx.foreach(S.score_rows, score_row)),
                variant="ghost", width="100%"),
            empty_hint("Scores appear after the first run.")))


def why_card() -> rx.Component:
    return card(
        "Why This Model?", "lightbulb",
        rx.cond(
            S.has_result,
            rx.vstack(rx.foreach(S.reasons, lambda t: rx.hstack(
                rx.icon("circle-check", size=18, color="var(--green-9)", flex_shrink="0"),
                rx.text(t, size="2"), align="start", spacing="2")), spacing="2", align="start"),
            empty_hint("The checklist explains each routing choice.")))


def analysis_card() -> rx.Component:
    return card(
        "Prompt Analysis", "file-text",
        rx.cond(
            S.has_result,
            rx.vstack(rx.foreach(S.analysis, lambda a: rx.hstack(
                rx.text(a["label"], size="2", width="90px", color="var(--slate-11)"),
                rx.badge(a["value"], color_scheme=a["color"], variant="soft"),
                align="center")), spacing="2", align="start"),
            empty_hint("Detected prompt features show up here.")))


def summary_card() -> rx.Component:
    return card(
        "Quick Summary", "chart-column",
        rx.hstack(
            rx.text(rx.cond(S.has_result | (S.summary != ""), S.summary, "Run the router to get a summary."),
                    size="2", color="var(--slate-11)", flex="1"),
            rx.cond(S.has_result, rx.badge(S.selected_tier, color_scheme=S.selected_color, size="3",
                                           variant="soft", radius="full"),),
            align="center", width="100%"))


# ---- Pages -------------------------------------------------------------------
def home_view() -> rx.Component:
    return rx.vstack(
        rx.grid(
            rx.vstack(prompt_card(), decision_card(), responses_card(), spacing="4", width="100%"),
            rx.vstack(pool_card(), scores_card(), why_card(), analysis_card(), spacing="4", width="100%"),
            columns=rx.breakpoints(initial="1", lg="2fr 1fr"), spacing="4", width="100%", align="start"),
        summary_card(), spacing="4", width="100%")


def pool_view() -> rx.Component:
    return rx.vstack(pool_card(), card("Model details", "boxes", empty_hint(
        "Placeholder: add latency / price / quality statistics per tier here.")), spacing="4", width="100%")


def settings_view() -> rx.Component:
    return card("Settings", "settings", rx.text(
        "Placeholder: put API-key status, λ (cost weight), agent selection (Q-learning / DQN) and "
        "state-feature choice (handcrafted / embedding) here.", size="2", color="var(--slate-10)"))


def about_view() -> rx.Component:
    return card("About", "info", rx.text(
        "An RL agent learns to send each prompt to the cheapest model that still answers well. "
        "Reward = quality − λ · normalised cost.", size="2", color="var(--slate-10)"))


# ==========================================================================
# 4. PAGE + APP
# ==========================================================================
def index() -> rx.Component:
    return rx.box(
        header(),
        rx.hstack(
            sidebar(),
            rx.box(
                rx.match(S.active_nav, ("Home", home_view()), ("Model Pool", pool_view()),
                         ("Settings", settings_view()), about_view()),
                padding="24px", flex="1", min_width="0"),
            spacing="0", align="start", width="100%"),
        background="var(--slate-2)", min_height="100vh")


# Theme is configured in rxconfig.py (RadixThemesPlugin, Reflex >= 0.9).
app = rx.App()
app.add_page(index, route="/", title="LLM Router RL", on_load=S.init_page)
