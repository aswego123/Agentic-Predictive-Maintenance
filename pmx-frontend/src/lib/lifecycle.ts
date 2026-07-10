import type { CycleSnapshot } from '@/types/domain';

/**
 * Client-side port of `predictive-management.py::_generate_recommendations`
 * + scheduling logic. Given the physics-prediction health_status and the
 * remaining RUL in hours, produce:
 *   - a predicted failure calendar date
 *   - a "schedule maintenance by" date (with urgency band + rationale)
 *   - a "latest safe operation" date
 *   - human-readable recommendation strings (emoji-decorated to match the CLI)
 *
 * Everything is pure — no state, no fetch. Feeds the LifecyclePredictionPanel.
 */

export type UrgencyBand = 'immediate' | 'urgent' | 'planned' | 'routine' | 'none';

export interface MaintenanceSchedule {
  urgency: UrgencyBand;
  urgencyLabel: string;
  scheduleByDate: Date | null;   // "schedule maintenance by" this date
  scheduleByDays: number | null; // days from now
  latestSafeDate: Date | null;   // "latest safe operation" boundary
  predictedFailureDate: Date | null;
  windowDescription: string;     // e.g. "within 7 days"
  rationale: string;
  recommendations: string[];
}

const HOURS_PER_DAY = 24;

const URGENCY_META: Record<UrgencyBand, { label: string; window: string; rationale: string }> = {
  immediate: {
    label: 'Immediate',
    window: 'Now — do not operate',
    rationale: 'Component failed or effectively at end-of-life. Operating further risks catastrophic failure.',
  },
  urgent: {
    label: 'Urgent',
    window: 'Within 7 days',
    rationale: 'Critical health status. RUL is short enough that scheduled downtime must be arranged this week.',
  },
  planned: {
    label: 'Planned (2–4 weeks)',
    window: 'Within 2–4 weeks',
    rationale: 'Warning-level degradation detected. Book maintenance into the next planning cycle.',
  },
  routine: {
    label: 'Routine inspection',
    window: 'At next scheduled inspection',
    rationale: 'Health is normal or monitoring-band. Fold into the next routine visit — no schedule change needed.',
  },
  none: {
    label: 'No action',
    window: 'N/A',
    rationale: 'Prediction is unavailable for this cycle.',
  },
};

function daysFromHours(h: number) {
  return h / HOURS_PER_DAY;
}

function addHours(date: Date, hours: number) {
  return new Date(date.getTime() + hours * 3600 * 1000);
}

export function computeSchedule(cycle: CycleSnapshot): MaintenanceSchedule {
  const physics = (cycle.physics_prediction ?? {}) as Record<string, any>;
  const rulHours = Number(physics.rul_hours ?? NaN);
  const healthStatus = String(physics.health_status ?? '').toLowerCase();
  const failureMode = String(physics.failure_mode ?? 'wear').toLowerCase();
  const now = new Date();

  // -----------------------------------------------------------------
  // Prefer values already produced by prediction-managent.py
  // -----------------------------------------------------------------
  // The physics engine wrapper forwards the CLI's own outputs on the
  // `physics_prediction` blob:
  //   - `predicted_failure_date`  (ISO string, from FatigueAnalyzer)
  //   - `recommendations`         (emoji-decorated strings, from
  //                                `_generate_recommendations()`)
  // We use those verbatim when present so the UI stays in lockstep
  // with what the CLI would print. We only fall back to a local
  // recompute if the backend didn't ship them (e.g. legacy cycle).
  // -----------------------------------------------------------------
  const backendFailureDateStr = physics.predicted_failure_date as string | undefined;
  const backendFailureDate =
    backendFailureDateStr ? new Date(backendFailureDateStr) : null;
  const backendRecommendations = Array.isArray(physics.recommendations)
    ? (physics.recommendations as string[]).filter((s) => typeof s === 'string' && s.length > 0)
    : [];

  if (!Number.isFinite(rulHours) || rulHours <= 0) {
    return {
      urgency: 'none',
      urgencyLabel: URGENCY_META.none.label,
      scheduleByDate: null,
      scheduleByDays: null,
      latestSafeDate: null,
      predictedFailureDate: backendFailureDate,
      windowDescription: URGENCY_META.none.window,
      rationale: URGENCY_META.none.rationale,
      recommendations: backendRecommendations,
    };
  }

  const predictedFailureDate =
    backendFailureDate ?? addHours(now, rulHours);
  const latestSafeDate = predictedFailureDate;
  const daysRemaining = daysFromHours(rulHours);

  // -----------------------------------------------------------------
  // Urgency is decided ONLY from physics `health_status` — matches
  // prediction-managent.py::_generate_recommendations exactly.
  // Judge verdict + anomaly severity are shown separately (VerdictHero
  // + AnomalyPanel), so they still surface, but they no longer
  // override the maintenance schedule. Otherwise a cycle with
  // health_score=92 could show "Immediate maintenance" just because
  // the anomaly gate saw a vibration spike or the LLM was cautious.
  // -----------------------------------------------------------------
  let urgency: UrgencyBand;
  switch (healthStatus) {
    case 'failed':
      urgency = 'immediate';
      break;
    case 'critical':
      urgency = 'urgent';
      break;
    case 'warning':
      urgency = 'planned';
      break;
    case 'monitoring':
    case 'normal':
    case '':
    default:
      urgency = 'routine';
      break;
  }

  // RUL-based escalation. A safe stress ratio (health_status=normal or
  // monitoring) doesn't rescue a part that's already at end-of-life —
  // if remaining life is measured in days, the schedule must reflect
  // that regardless of what the stress-based health band says.
  //
  //   RUL < 2 days   → immediate (do not operate)
  //   RUL < 7 days   → at least urgent
  //   RUL < 28 days  → at least planned
  //
  // We only *raise* urgency here; nothing lowers a critical/failed
  // health band just because RUL happens to look long.
  const URGENCY_ORDER: UrgencyBand[] = ['none', 'routine', 'planned', 'urgent', 'immediate'];
  const escalate = (a: UrgencyBand, b: UrgencyBand): UrgencyBand =>
    URGENCY_ORDER.indexOf(b) > URGENCY_ORDER.indexOf(a) ? b : a;

  if (daysRemaining < 2) urgency = escalate(urgency, 'immediate');
  else if (daysRemaining < 7) urgency = escalate(urgency, 'urgent');
  else if (daysRemaining < 28) urgency = escalate(urgency, 'planned');

  // Schedule-by-date bands match the CLI's recommendation text:
  //   FAILED   → 0 days      ("Immediate replacement")
  //   CRITICAL → up to 7d    ("URGENT: Schedule maintenance immediately")
  //   WARNING  → 14–28d      ("Schedule maintenance within 2-4 weeks")
  //   MONIT/NORMAL → 30–180d ("Schedule next inspection in {days} days")
  let scheduleByDays: number;
  switch (urgency) {
    case 'immediate':
      scheduleByDays = 0;
      break;
    case 'urgent':
      scheduleByDays = Math.min(7, Math.max(1, Math.floor(daysRemaining * 0.25)));
      break;
    case 'planned':
      scheduleByDays = Math.min(28, Math.max(14, Math.floor(daysRemaining * 0.5)));
      break;
    case 'routine':
    default:
      scheduleByDays = Math.min(180, Math.max(30, Math.floor(daysRemaining * 0.1)));
      break;
  }

  // Absolute safety clamp: the schedule-by date must never be later
  // than the predicted failure date. This was the source of the bug
  // where a healthy-looking stress band showed "schedule in ~1 month"
  // while RUL was 5 days — a nonsensical schedule for the operator.
  if (daysRemaining > 0) {
    const latestSafeDays = Math.max(0, Math.floor(daysRemaining - 1));
    scheduleByDays = Math.min(scheduleByDays, latestSafeDays);
  }

  const scheduleByDate = addHours(now, scheduleByDays * HOURS_PER_DAY);

  // Recommendations: use the ones the CLI already produced when the
  // backend shipped them; otherwise regenerate locally so cycles from
  // old snapshots (or the deterministic fallback path) still render.
  const recommendations =
    backendRecommendations.length > 0
      ? backendRecommendations
      : buildRecommendations(healthStatus, daysRemaining, failureMode);

  const meta = URGENCY_META[urgency];
  return {
    urgency,
    urgencyLabel: meta.label,
    scheduleByDate,
    scheduleByDays,
    latestSafeDate,
    predictedFailureDate,
    windowDescription: meta.window,
    rationale: meta.rationale,
    recommendations,
  };
}

/**
 * Verbatim port of `_generate_recommendations` from prediction-managent.py
 * so the CLI output and the UI stay in lockstep.
 */
function buildRecommendations(
  status: string,
  remainingDays: number,
  failureMode: string,
): string[] {
  const daysStr = Math.max(0, Math.floor(remainingDays));
  switch (status) {
    case 'normal':
      return [
        '✅ Continue normal operations',
        `📅 Schedule next inspection in ${daysStr} days`,
        '📊 Monitor sensor data regularly',
      ];
    case 'monitoring':
      return [
        '🔍 Increase monitoring frequency',
        `📅 Schedule inspection in ${daysStr} days`,
        '📊 Track degradation trends',
        '🔧 Prepare maintenance plan',
      ];
    case 'warning':
      return [
        '⚠️ Schedule maintenance within 2–4 weeks',
        `🔧 Inspect for ${failureMode}`,
        `📅 Latest safe operation: ${daysStr} days`,
        '📊 Increase monitoring frequency',
        '🔬 Perform NDT inspection',
      ];
    case 'critical':
      return [
        '🚨 URGENT: Schedule maintenance immediately',
        `🔧 Address ${failureMode}`,
        `⚠️ Maximum safe time: ${daysStr} days`,
        '📊 Real-time monitoring required',
        '🔬 Comprehensive NDT inspection',
      ];
    case 'failed':
      return [
        '⛔ COMPONENT FAILED — Immediate replacement',
        '🔧 Replace component before operation',
        '🔍 Investigate root cause',
        '📊 Update maintenance procedures',
      ];
    default:
      return [];
  }
}

export function fmtCalendarDate(d: Date | null | undefined): string {
  if (!d) return '—';
  return d.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    weekday: 'short',
  });
}

export function fmtRelativeDays(d: Date | null | undefined): string {
  if (!d) return '—';
  const days = Math.round((d.getTime() - Date.now()) / (1000 * 3600 * 24));
  if (days === 0) return 'today';
  if (days < 0) return `${Math.abs(days)}d overdue`;
  if (days < 30) return `in ${days}d`;
  const months = Math.round(days / 30);
  if (months < 24) return `in ~${months} months`;
  const years = (days / 365).toFixed(1);
  return `in ~${years} years`;
}

export function fmtDurationHours(h: number | null | undefined): string {
  if (h === null || h === undefined || Number.isNaN(h)) return '—';
  if (!Number.isFinite(h)) return '∞';
  if (h < 48) return `${h.toFixed(1)} h`;
  const days = h / 24;
  if (days < 60) return `${days.toFixed(1)} days`;
  const years = days / 365.25;
  if (years < 1) return `${(days / 30).toFixed(1)} months`;
  return `${years.toFixed(1)} years`;
}
