"""Smart Water Impact Tracker Streamlit app."""
from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import altair as alt
import pandas as pd
import streamlit as st

DATA_ZIP_PATH = Path(__file__).resolve().parent / "clean_data" / "clean_data.zip"
METER_READS_PATH = "clean_data/meter_reads.csv"
HOUSEHOLD_META_PATH = "clean_data/houshold_meta_data.csv"
APPLIANCES_PATH = "clean_data/appliances.csv"

TOWN_LABELS = {"T": "Terre di Pedemonte", "V": "Valencia"}


@dataclass
class HouseholdSnapshot:
    smart_meter_id: str
    total_consumption: float
    avg_daily_consumption: float
    active_days: int
    recent_trend: float
    percentile_rank: float
    impact_score: int
    status_label: str
    town: str
    per_capita_consumption: float | None
    town_percentile_rank: float | None


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    """Load and preprocess datasets from the zipped archive."""
    if not DATA_ZIP_PATH.exists():
        st.error("Unable to locate clean data archive. Please ensure clean_data.zip is present.")
        st.stop()

    with zipfile.ZipFile(DATA_ZIP_PATH) as archive:
        with archive.open(METER_READS_PATH) as fh:
            meter_reads = pd.read_csv(fh, parse_dates=["timestamp"])
        with archive.open(HOUSEHOLD_META_PATH) as fh:
            household_meta = pd.read_csv(fh)
        with archive.open(APPLIANCES_PATH) as fh:
            appliances = pd.read_csv(fh)

    # Clean household metadata
    household_meta = household_meta.replace("NA", pd.NA)
    numeric_cols = [
        "household_size",
        "household_garden_area",
        "household_pool_volume",
        "number_bathrooms",
        "building_size",
    ]
    for col in numeric_cols:
        household_meta[col] = pd.to_numeric(household_meta[col], errors="coerce")

    bool_like_cols = ["household_pool", "household_garden", "irrigation_system", "ecomode", "timer"]
    for col in bool_like_cols:
        household_meta[col] = household_meta[col].map({1: True, 0: False, "1": True, "0": False})

    # Prepare appliance data
    appliances["number"] = pd.to_numeric(appliances["number"], errors="coerce").fillna(0)

    # Prepare meter reads with consumption deltas
    meter_reads = meter_reads.dropna(subset=["smart_meter_id", "meter_reading"])
    meter_reads = meter_reads.sort_values(["smart_meter_id", "timestamp"])
    meter_reads["meter_reading"] = pd.to_numeric(meter_reads["meter_reading"], errors="coerce")
    meter_reads["consumption"] = (
        meter_reads.groupby("smart_meter_id")["meter_reading"].diff().fillna(0).clip(lower=0)
    )
    meter_reads["date"] = meter_reads["timestamp"].dt.floor("D")

    daily_consumption = (
        meter_reads.groupby(["smart_meter_id", "date"], as_index=False)["consumption"].sum()
    )

    city_daily = daily_consumption.groupby("date", as_index=False)["consumption"].mean()
    household_totals = daily_consumption.groupby("smart_meter_id").agg(
        total_consumption=("consumption", "sum"),
        avg_daily_consumption=("consumption", "mean"),
        active_days=("date", "nunique"),
    )

    recent_window = 30
    recent_totals = pd.Series(dtype=float)
    if not daily_consumption.empty:
        max_date = daily_consumption["date"].max()
        window_start = max_date - pd.Timedelta(days=recent_window - 1)
        recent_daily = daily_consumption[daily_consumption["date"].between(window_start, max_date)]
        recent_totals = recent_daily.groupby("smart_meter_id")["consumption"].mean()
    household_totals["recent_avg"] = recent_totals

    city_distribution = household_totals["avg_daily_consumption"].dropna().rename("avg_daily_consumption")

    appliances_pivot = appliances.pivot_table(
        index="smart_meter_id",
        columns="appliance",
        values="number",
        aggfunc="sum",
        fill_value=0,
    )

    household_profile = (
        household_meta.drop(columns=["appliance"], errors="ignore")
        .drop_duplicates(subset="smart_meter_id", keep="first")
        .set_index("smart_meter_id")
    )

    household_totals = household_totals.join(household_profile, how="left")
    household_totals["town"] = (
        household_totals.index.to_series().str[0].map(TOWN_LABELS).fillna("Unknown")
    )
    with pd.option_context("mode.use_inf_as_na", True):
        household_totals["per_capita_consumption"] = (
            household_totals["avg_daily_consumption"]
            / household_totals["household_size"].replace({0: pd.NA})
        )

    daily_consumption["town"] = daily_consumption["smart_meter_id"].str[0].map(TOWN_LABELS)
    city_daily_by_town = (
        daily_consumption.dropna(subset=["town"]).groupby(["town", "date"], as_index=False)["consumption"].mean()
    )

    per_capita_series = household_totals["per_capita_consumption"].dropna()
    city_metrics = {
        "city_avg_daily": float(household_totals["avg_daily_consumption"].mean()),
        "city_avg_per_capita": float(per_capita_series.mean()) if not per_capita_series.empty else math.nan,
        "avg_by_town": household_totals.groupby("town")["avg_daily_consumption"].mean(),
        "per_capita_by_town": household_totals.groupby("town")["per_capita_consumption"].mean(),
    }

    return {
        "meter_reads": meter_reads,
        "daily_consumption": daily_consumption,
        "city_daily": city_daily,
        "city_daily_by_town": city_daily_by_town,
        "household_totals": household_totals,
        "city_distribution": city_distribution,
        "household_meta": household_meta,
        "household_profile": household_profile,
        "appliances": appliances,
        "appliance_matrix": appliances_pivot,
        "city_metrics": city_metrics,
    }


def build_snapshot(
    smart_meter_id: str, data: dict[str, pd.DataFrame],
) -> HouseholdSnapshot | None:
    household_totals = data["household_totals"]
    if smart_meter_id not in household_totals.index:
        return None

    city_totals = household_totals["total_consumption"].dropna()
    selected = household_totals.loc[smart_meter_id]

    percentile_rank = 0.0
    if not city_totals.empty:
        rank = (city_totals < selected["total_consumption"]).sum()
        percentile_rank = (rank / len(city_totals)) * 100

    impact_score = int(round(100 - percentile_rank))

    if impact_score >= 80:
        status_label = "🌟 Trailblazer"
    elif impact_score >= 55:
        status_label = "✅ On Track"
    else:
        status_label = "⚠️ Needs Attention"

    recent_trend = float(selected.get("recent_avg", float("nan")))

    town = selected.get("town", "Unknown")
    per_capita = selected.get("per_capita_consumption")
    if isinstance(per_capita, pd.Series):
        per_capita = per_capita.iloc[0]
    per_capita_value = float(per_capita) if pd.notna(per_capita) else None

    town_percentile_rank = None
    if town and town != "Unknown":
        town_totals = household_totals[household_totals["town"] == town]["total_consumption"].dropna()
        if not town_totals.empty:
            rank = (town_totals < selected["total_consumption"]).sum()
            town_percentile_rank = (rank / len(town_totals)) * 100

    return HouseholdSnapshot(
        smart_meter_id=smart_meter_id,
        total_consumption=float(selected["total_consumption"]),
        avg_daily_consumption=float(selected["avg_daily_consumption"]),
        active_days=int(selected["active_days"]),
        recent_trend=recent_trend,
        percentile_rank=percentile_rank,
        impact_score=impact_score,
        status_label=status_label,
        town=str(town),
        per_capita_consumption=per_capita_value,
        town_percentile_rank=town_percentile_rank,
    )


def render_profile(snapshot: HouseholdSnapshot, data: dict[str, pd.DataFrame]) -> None:
    household_profile = data["household_profile"]
    daily_consumption = data["daily_consumption"]
    city_daily = data["city_daily"]
    appliance_matrix = data["appliance_matrix"]
    city_metrics = data["city_metrics"]

    profile = None
    if snapshot.smart_meter_id in household_profile.index:
        profile = household_profile.loc[snapshot.smart_meter_id]

    st.subheader("Your Impact Snapshot")
    metrics = st.columns(4)

    delta_recent = None
    if not math.isnan(snapshot.recent_trend):
        delta_recent = f"{snapshot.recent_trend - snapshot.avg_daily_consumption:+.1f} L vs 30-day"
    metrics[0].metric("Avg daily use", f"{snapshot.avg_daily_consumption:.1f} L", delta=delta_recent)

    city_avg_daily = city_metrics.get("city_avg_daily", math.nan)
    if math.isnan(city_avg_daily):
        metrics[1].metric("City benchmark", "—")
    else:
        delta_vs_city = snapshot.avg_daily_consumption - city_avg_daily
        metrics[1].metric("City benchmark", f"{city_avg_daily:.1f} L", delta=f"{delta_vs_city:+.1f} L vs you")

    per_capita_text = "—"
    per_capita_delta = None
    city_per_capita = city_metrics.get("city_avg_per_capita", math.nan)
    if snapshot.per_capita_consumption is not None:
        per_capita_text = f"{snapshot.per_capita_consumption:.1f} L"
        if not math.isnan(city_per_capita):
            per_capita_delta = f"{snapshot.per_capita_consumption - city_per_capita:+.1f} L vs city"
    metrics[2].metric("Per person use", per_capita_text, delta=per_capita_delta)

    metrics[3].metric("Impact score", snapshot.impact_score, delta=snapshot.status_label)

    community_cols = st.columns(3)
    city_position = max(0.0, 100 - snapshot.percentile_rank)
    community_cols[0].metric("City standing", f"Top {city_position:.0f}%")
    if snapshot.town_percentile_rank is not None:
        town_position = max(0.0, 100 - snapshot.town_percentile_rank)
        community_cols[1].metric(f"{snapshot.town} standing", f"Top {town_position:.0f}%")
    else:
        community_cols[1].metric("Local standing", "—")
    community_cols[2].metric("Active days tracked", snapshot.active_days)

    with st.expander("Profile & Efficiency Signals", expanded=True):
        cols = st.columns(3)
        if profile is not None:
            cols[0].write(
                f"**Household size:** {profile.get('household_size', '—') or '—'}\n\n"
                f"**Residency type:** {profile.get('residency_type', '—')}\n\n"
                f"**Bathrooms:** {profile.get('number_bathrooms', '—') or '—'}"
            )
            cols[1].write(
                f"**Building size:** {profile.get('building_size', '—') or '—'} m²\n\n"
                f"**Garden area:** {profile.get('household_garden_area', '—') or '—'} m²\n\n"
                f"**Pool volume:** {profile.get('household_pool_volume', '—') or '—'} L"
            )
            eco_flags = []
            efficiency = profile.get("efficiency")
            if efficiency and efficiency is not pd.NA:
                eco_flags.append(f"🏷️ Dominant appliance efficiency: {efficiency}")
            if bool(profile.get("household_pool")):
                eco_flags.append("🏊 Pool present")
            if bool(profile.get("household_garden")):
                eco_flags.append("🌱 Garden to irrigate")
            if bool(profile.get("irrigation_system")):
                eco_flags.append("💧 Smart irrigation installed")
            if bool(profile.get("ecomode")):
                eco_flags.append("⚙️ Eco mode enabled")
            if bool(profile.get("timer")):
                eco_flags.append("⏱️ Smart timers active")
            cols[2].write("\n".join(eco_flags) or "No efficiency flags recorded.")
        else:
            cols[0].info("Metadata not available for this household.")

    household_daily = daily_consumption[daily_consumption["smart_meter_id"] == snapshot.smart_meter_id]
    comparison = household_daily.merge(city_daily, on="date", how="left", suffixes=("_you", "_city"))

    if not comparison.empty:
        comparison_long = comparison.melt(
            id_vars="date",
            value_vars=["consumption_you", "consumption_city"],
            var_name="series",
            value_name="litres",
        )
        comparison_long["series"] = comparison_long["series"].map(
            {
                "consumption_you": "You",
                "consumption_city": "Smart city average",
            }
        )
        comparison_chart = (
            alt.Chart(comparison_long)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("litres:Q", title="Litres per day"),
                color=alt.Color("series:N", title=""),
                tooltip=["date:T", "series:N", alt.Tooltip("litres:Q", title="Litres", format=",.1f")],
            )
            .properties(height=320)
        )
        st.subheader("Impact trajectory")
        st.altair_chart(comparison_chart, use_container_width=True)

    city_distribution = data["city_distribution"]
    if not city_distribution.empty:
        distribution_df = city_distribution.to_frame().reset_index(drop=True)
        distribution_chart = (
            alt.Chart(distribution_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "avg_daily_consumption:Q",
                    title="Average daily consumption (L)",
                    bin=alt.Bin(maxbins=30),
                ),
                y=alt.Y("count()", title="Number of households"),
            )
            .properties(height=260)
        )
        your_marker = alt.Chart(pd.DataFrame({"avg_daily_consumption": [snapshot.avg_daily_consumption]})).mark_rule(
            color="orange"
        ).encode(x="avg_daily_consumption:Q")
        st.subheader("Where you stand citywide")
        st.altair_chart(distribution_chart + your_marker, use_container_width=True)

    st.subheader("Gamified progress")
    streak = calculate_positive_streak(comparison)
    col1, col2 = st.columns(2)
    col1.metric("Days better than city", f"{streak} day(s)")
    col2.progress(min(snapshot.impact_score, 100) / 100.0, text=f"Impact score: {snapshot.impact_score}")

    appliances_row = (
        appliance_matrix.loc[[snapshot.smart_meter_id]]
        if snapshot.smart_meter_id in appliance_matrix.index
        else None
    )
    with st.expander("Household fixtures & focus areas", expanded=False):
        if appliances_row is not None:
            long_appliances = (
                appliances_row.T.reset_index().rename(
                    columns={snapshot.smart_meter_id: "count", "index": "appliance"}
                )
            )
            long_appliances = long_appliances[long_appliances["count"] > 0]
            if not long_appliances.empty:
                appliances_chart = (
                    alt.Chart(long_appliances)
                    .mark_bar()
                    .encode(
                        x=alt.X("count:Q", title="Number of fixtures"),
                        y=alt.Y("appliance:N", sort="-x", title=""),
                        tooltip=["appliance", "count"],
                    )
                    .properties(height=260)
                )
                st.altair_chart(appliances_chart, use_container_width=True)
            st.dataframe(long_appliances.set_index("appliance"), use_container_width=True)
        else:
            st.info("No appliance records available for this household.")

    st.subheader("Impact boosters")
    for tip in build_action_plan(snapshot, profile, comparison, city_metrics):
        st.markdown(f"- {tip}")


def calculate_positive_streak(comparison: pd.DataFrame) -> int:
    if comparison.empty:
        return 0
    comparison = comparison.dropna(subset=["consumption_you", "consumption_city"]).sort_values("date")
    if comparison.empty:
        return 0
    mask = comparison["consumption_you"] <= comparison["consumption_city"]
    streak = 0
    for flag in mask[::-1]:
        if flag:
            streak += 1
        else:
            break
    return streak


def build_action_plan(
    snapshot: HouseholdSnapshot,
    profile: pd.Series | None,
    comparison: pd.DataFrame,
    city_metrics: dict,
) -> List[str]:
    tips: List[str] = []

    def _to_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    if profile is not None:
        if bool(profile.get("household_pool")):
            tips.append(
                "Pool detected: cover and maintain filtration schedules to cut evaporation losses by up to 50%."
            )
        garden_area = _to_float(profile.get("household_garden_area"))
        if bool(profile.get("household_garden")) and not bool(profile.get("irrigation_system")):
            tips.append("Garden without smart irrigation: consider drip systems or night-time watering to save litres.")
        if garden_area and not math.isnan(garden_area) and garden_area > 150:
            tips.append("Large garden: schedule evening watering quests to trim evaporation losses.")
        if profile.get("efficiency") in {"A", "B", pd.NA, None}:
            tips.append("Upgrade washing appliances to A++ or better for immediate water savings.")
        if not bool(profile.get("timer")):
            tips.append("Install tap or shower timers to gamify shorter usage windows.")
        building_size = _to_float(profile.get("building_size"))
        city_per_capita = city_metrics.get("city_avg_per_capita", math.nan)
        if (
            snapshot.per_capita_consumption is not None
            and not math.isnan(city_per_capita)
            and snapshot.per_capita_consumption > city_per_capita
            and building_size
            and building_size > 150
        ):
            tips.append(
                "Spacious home challenge: audit bathroom fixtures for aerators to cut per-person use below the city average."
            )

    if not comparison.empty:
        recent = comparison.tail(7)
        if {"consumption_you", "consumption_city"}.issubset(recent.columns):
            recent_gap = recent["consumption_you"].mean() - recent["consumption_city"].mean()
            if recent_gap > 0:
                tips.append(
                    "You are above the city average this week—try a 7-day challenge to shave off "
                    f"{recent_gap:.0f} L per day and unlock a new badge."
                )
            else:
                tips.append("You are beating the city average this week—share your tips to earn community kudos!")

    if snapshot.impact_score >= 90:
        tips.append("Invite neighbours to join a conservation league and multiply the community impact.")
    elif snapshot.impact_score < 55:
        tips.append("Unlock the 'Reset' quest: set a goal to beat the city average three days in a row for a bronze badge.")

    if not tips:
        tips.append("Keep tracking daily to unlock tailored conservation quests.")
    return tips


def render_overview(data: dict[str, pd.DataFrame], focus_town: str | None) -> None:
    st.title("Smart Water Impact Tracker")
    st.caption(
        "Compare your household's water journey with the smart city benchmark, unlock gamified goals, and discover actionable insights."
    )

    household_totals = data["household_totals"].copy()
    city_metrics = data["city_metrics"]

    scope_label = "All towns"
    if focus_town and focus_town != "All":
        household_totals = household_totals[household_totals["town"] == focus_town]
        scope_label = focus_town

    if household_totals.empty:
        st.info("No households found for the selected scope. Adjust the filters to explore the data.")
        return

    household_totals = household_totals.sort_values("avg_daily_consumption")

    if focus_town and focus_town != "All":
        city_daily_focus = data["city_daily_by_town"]
        city_daily_focus = city_daily_focus[city_daily_focus["town"] == focus_town][["date", "consumption"]]
        avg_daily = city_metrics["avg_by_town"].get(focus_town, math.nan)
        avg_per_capita = city_metrics["per_capita_by_town"].get(focus_town, math.nan)
    else:
        city_daily_focus = data["city_daily"]
        avg_daily = city_metrics.get("city_avg_daily", math.nan)
        avg_per_capita = city_metrics.get("city_avg_per_capita", math.nan)

    col1, col2, col3, col4 = st.columns(4)
    if not math.isnan(avg_daily):
        col1.metric(f"{scope_label} avg daily use", f"{avg_daily:.1f} L")
    else:
        col1.metric(f"{scope_label} avg daily use", "—")

    col2.metric("Households tracked", len(household_totals))

    if not math.isnan(avg_per_capita):
        col3.metric(f"{scope_label} per person", f"{avg_per_capita:.1f} L")
    else:
        col3.metric(f"{scope_label} per person", "—")

    top_impact = household_totals.head(1)
    col4.metric(
        "Best performer",
        top_impact.index[0],
        delta=f"{top_impact['avg_daily_consumption'].iloc[0]:.1f} L/day",
    )

    if not city_daily_focus.empty:
        trend_chart = (
            alt.Chart(city_daily_focus)
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("consumption:Q", title="Average litres per day"),
            )
            .properties(height=260)
        )
        st.subheader(f"{scope_label} benchmark trend")
        st.altair_chart(trend_chart, use_container_width=True)

    if focus_town in (None, "All"):
        town_summary = city_metrics["avg_by_town"].dropna()
        if not town_summary.empty:
            town_chart = (
                alt.Chart(town_summary.reset_index())
                .mark_bar()
                .encode(
                    x=alt.X("avg_daily_consumption:Q", title="Avg daily consumption (L)"),
                    y=alt.Y("town:N", title="Town"),
                    tooltip=["town", alt.Tooltip("avg_daily_consumption:Q", format=",.1f")],
                )
                .properties(height=160)
            )
            st.subheader("Town comparison")
            st.altair_chart(town_chart, use_container_width=True)

    leaderboard = (
        household_totals
        .assign(
            impact_score=lambda df: (1 - df["avg_daily_consumption"].rank(pct=True)) * 100,
        )
        .head(10)
        .reset_index()
    )
    leaderboard_display = leaderboard[
        ["smart_meter_id", "town", "avg_daily_consumption", "per_capita_consumption", "impact_score"]
    ]
    leaderboard_display.columns = [
        "Smart meter",
        "Town",
        "Avg daily (L)",
        "Per person (L)",
        "Impact score",
    ]
    st.subheader("Top water savers")
    st.dataframe(leaderboard_display, use_container_width=True)


def main() -> None:
    data = load_data()

    totals = data["household_totals"]
    town_choices = sorted(t for t in totals["town"].dropna().unique())
    st.sidebar.subheader("City scope")
    focus_town = st.sidebar.selectbox("Focus town", ["All", *town_choices], index=0)

    st.sidebar.divider()

    if focus_town and focus_town != "All":
        smart_scope = totals[totals["town"] == focus_town].index.tolist()
    else:
        smart_scope = totals.index.tolist()

    smart_ids = sorted(smart_scope)
    st.sidebar.header("Find My Smart ID")
    query = st.sidebar.text_input("Search by ID", placeholder="e.g. T204")
    filtered_ids = [sid for sid in smart_ids if query.lower() in sid.lower()] if query else smart_ids
    selected_id = None
    if filtered_ids:
        selected_id = st.sidebar.selectbox("Select your smart meter", filtered_ids)
    else:
        st.sidebar.warning("No smart IDs match your search.")

    if selected_id:
        render_overview(data, focus_town)
        st.divider()
        snapshot = build_snapshot(selected_id, data)
        if snapshot is None:
            st.warning("No usage data found for this smart meter.")
            return
        st.header(f"Household {selected_id}")
        render_profile(snapshot, data)
    else:
        render_overview(data, focus_town)
        st.info("Use the sidebar to search for your smart meter ID and unlock personalised insights.")


if __name__ == "__main__":
    main()
