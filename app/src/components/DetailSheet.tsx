import { Drawer } from "vaul";
import type { Alert, Window } from "../types";
import { BAND_KO, DIR_ICON, TOUCH_ICON, fmtPct, type PEvent } from "../lib/prepare";
import { useReplay } from "../store";

const LONG_MEANING: Record<string, string> = {
  UP_ALERT: "가격이 오를 만한 조건이 모였다는 규칙 신호예요.",
  UP_CONTINUATION: "오르던 흐름이 이어질 만하다는 규칙 신호예요.",
  RISK_ALERT: "아래로 밀릴 위험이 커졌다는 규칙 신호예요.",
  WEAKENING_ALERT: "오르던 힘이 약해지고 있다는 규칙 신호예요.",
  VOLATILITY_ALERT: "가격이 크게 흔들리기 시작했다는 규칙 신호예요.",
};
const DOWN_MEANING: Record<string, string> = {
  DOWN_ALERT: "가격이 내려갈 만한 조건이 모였다는 규칙 신호예요.",
  DOWN_CONTINUATION: "내리던 흐름이 이어질 만하다는 규칙 신호예요.",
  SHORT_RISK_ALERT: "하방 쪽에서 보면, 반대로 튀어 오를 위험이 커졌다는 신호예요.",
  DOWN_WEAKENING_ALERT: "내리던 힘이 약해지고 있다는 규칙 신호예요.",
  VOLATILITY_ALERT: "가격이 크게 흔들리기 시작했다는 규칙 신호예요.",
};

export default function DetailSheet() {
  const data = useReplay((s) => s.data);
  const selected = useReplay((s) => s.selected);
  const idx = useReplay((s) => s.idx);
  const select = useReplay((s) => s.select);
  const resume = useReplay((s) => s.resumeAfterSheet);
  const e = data && selected !== null ? data.events[selected] : null;

  return (
    <Drawer.Root open={!!e} onOpenChange={(o) => !o && select(null)}>
      <Drawer.Portal>
        <Drawer.Overlay className="fixed inset-0 z-40 bg-black/45" />
        <Drawer.Content
          className="fixed inset-x-0 bottom-0 z-50 mx-auto flex max-h-[88dvh] max-w-xl flex-col rounded-t-3xl bg-surface text-ink outline-none"
          aria-describedby={undefined}
        >
          <div className="mx-auto mt-2.5 h-1.5 w-10 shrink-0 rounded-full bg-line" />
          {e && (
            <div
              className="min-h-0 overflow-y-auto px-5 pt-3"
              style={{ paddingBottom: "max(env(safe-area-inset-bottom), 20px)" }}
            >
              <Header e={e} />
              <div className="mt-4 space-y-4">
                {e.kind === "touch" ? (
                  <TouchMeaning e={e} />
                ) : (
                  e.alerts!.map((a, i) => <AlertMeaning key={i} a={a} />)
                )}
                <Outcome e={e} idx={idx} />
                <div>
                  <p className="mb-1.5 text-[13px] font-semibold text-muted">원본 로그 (프로그램이 실제로 찍은 줄)</p>
                  <pre className="overflow-x-auto rounded-xl bg-term p-3 font-mono text-[11.5px] leading-5 text-term-ink">
                    {e.raw}
                  </pre>
                </div>
              </div>
              <button
                onClick={() => select(null)}
                className="mt-5 h-12 w-full rounded-2xl bg-ink text-[16px] font-semibold text-surface"
              >
                {resume ? "닫고 계속 재생" : "닫기"}
              </button>
            </div>
          )}
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}

function Header({ e }: { e: PEvent }) {
  const decided = e.decision_time.slice(11, 16);
  const source =
    e.kind === "touch"
      ? "이평 돌파 · 5분선×20분선 + CNN 등급"
      : e.kind === "both"
        ? "롱 엔진과 하방 시스템이 같은 분에 울림"
        : e.kind === "long"
          ? "롱 엔진 · 규칙 + LightGBM 위험 점수"
          : "하방 시스템 · 규칙 + 반등 위험 점수";
  const title =
    e.kind === "touch"
      ? e.touch!.direction > 0
        ? "이평 상향 돌파"
        : "이평 하향 돌파"
      : e.alerts!.map((a) => a.type_ko).join(" + ");
  return (
    <div>
      <Drawer.Title className="text-[21px] font-bold leading-snug text-balance">
        {e.kind === "touch" ? (
          <span className="mr-1.5 text-touch">{TOUCH_ICON}</span>
        ) : (
          <span className={`mr-1.5 ${dirColor(e.alerts![0].dir)}`}>{DIR_ICON[e.alerts![0].dir]}</span>
        )}
        {title}
      </Drawer.Title>
      <p className="mt-1 text-[14px] text-muted">
        <span className="tabular">{e.bar}</span> 봉이 끝난 {decided}에 판단 · {source}
      </p>
    </div>
  );
}

const dirColor = (d: string) => (d === "up" ? "text-up" : d === "down" ? "text-down" : "text-neutral");

function AlertMeaning({ a }: { a: Alert }) {
  const long = a.side === "long";
  const meaning = (long ? LONG_MEANING : DOWN_MEANING)[a.type] ?? "";
  const score = a.prob === null ? "–" : a.prob.toFixed(2);
  const band = BAND_KO[a.band] ?? a.band;
  let agree = "";
  if (long && (a.type === "UP_ALERT" || a.type === "UP_CONTINUATION")) {
    if (a.band === "HIGH") agree = "규칙은 위를 봤지만 모델은 위험하다고 봤어요. 위험 필터가 일하는 장면이에요.";
    if (a.band === "LOW") agree = "규칙과 모델이 모두 괜찮다고 본 경우예요.";
  } else if (long && a.type === "RISK_ALERT" && a.band === "HIGH") {
    agree = "규칙과 모델이 모두 위험하다고 본 경우예요.";
  } else if (!long && (a.type === "DOWN_ALERT" || a.type === "DOWN_CONTINUATION")) {
    if (a.band === "HIGH") agree = "규칙은 아래를 봤지만 모델은 반등 위험이 크다고 봤어요.";
    if (a.band === "LOW") agree = "규칙과 모델이 모두 하방 쪽에 힘을 실은 경우예요.";
  }
  const grade = ["A", "B", "C"].includes(a.grade) ? a.grade : null;
  return (
    <div className="rounded-2xl bg-sunk p-4">
      <p className="text-[13px] font-semibold text-muted">
        {long ? "롱 엔진" : "하방 시스템"} · {a.type_ko}
        {grade && ` · ${grade}등급`}
      </p>
      <p className="mt-1.5 text-[16px] leading-relaxed">{meaning}</p>
      <div className="mt-3 rounded-xl bg-surface px-3 py-2.5">
        <p className="text-[12px] font-semibold text-muted">모델이 맞히려는 것</p>
        <p className="mt-0.5 text-[15px] font-semibold leading-snug">
          {long
            ? "지금부터 장 마감까지 −1.5% 이상 밀릴지"
            : "지금부터 장 마감까지 +1.5% 이상 튀어 오를지"}
        </p>
        <p className="mt-0.5 text-[12.5px] leading-relaxed text-muted">
          {long
            ? "오를지 내릴지를 맞히는 게 아니라, 지금 샀다면 −1.5% 손실 구간까지 내려갈 위험을 매긴 점수예요."
            : "내릴지를 맞히는 게 아니라, 하방으로 봤을 때 +1.5% 반대로 튀어 오를 위험을 매긴 점수예요."}
        </p>
      </div>
      <div className="mt-3 flex items-baseline gap-2">
        <span className="text-[13px] text-muted">{long ? "모델 위험 점수" : "모델 반등 위험 점수"}</span>
        <span className="tabular font-mono text-[18px] font-semibold">{score}</span>
        <span
          className={`text-[14px] font-semibold ${
            a.band === "HIGH" ? "text-risk-high" : a.band === "LOW" ? "text-risk-low" : "text-muted"
          }`}
        >
          {band}
        </span>
      </div>
      <p className="mt-1 text-[13px] leading-relaxed text-muted">
        0.60 이상은 높음, 0.45 이상은 보통이에요. 정확한 확률이 아니라 위험한 순서를 매긴 점수예요.
      </p>
      {agree && <p className="mt-2 text-[14px] font-medium leading-relaxed">{agree}</p>}
      {grade && (
        <p className="mt-2 text-[13px] text-muted">등급은 A가 조건이 가장 잘 맞은 경우, C가 가장 약한 경우예요.</p>
      )}
      {a.reasons.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {a.reasons.map((r) => (
            <span key={r} className="rounded-full bg-surface px-2.5 py-1 text-[12.5px] font-medium">
              {r}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function TouchMeaning({ e }: { e: PEvent }) {
  const t = e.touch!;
  const up = t.direction > 0;
  const bandKo = { near: "약함 (0.1% 이하)", mid: "중간", far: "강함 (0.3% 이상)" }[t.band];
  let gradeText: string;
  if (t.grade) {
    gradeText = `CNN 등급 ${t.grade}: 과거 같은 등급의 돌파 ${(t.grade_n ?? 0).toLocaleString(
      "ko-KR",
    )}건 중 ${Math.round((t.grade_rate ?? 0) * 100)}%가 10분 안에 ${up ? "+" : "−"}0.5%에 닿았어요. 전체 평균은 약 16%예요.`;
  } else if (t.status.startsWith("워밍업")) {
    gradeText = "장 초반이라 모델 입력(직전 60분 흐름)이 아직 다 모이지 않아 등급을 매기지 않았어요.";
  } else if (t.status.startsWith("창 불완전")) {
    gradeText = "장 마감이 가까워 10분 결과를 확인할 수 없어서 등급을 매기지 않았어요.";
  } else {
    gradeText = "이 돌파에는 등급이 없어요.";
  }
  return (
    <div className="rounded-2xl bg-sunk p-4">
      <p className="text-[13px] font-semibold text-muted">이평 돌파 알림</p>
      <p className="mt-1.5 text-[16px] leading-relaxed">
        5분 이동평균이 20분 이동평균을 {up ? "위로" : "아래로"} 넘었어요. 돌파 거리는 {bandKo}이에요.
      </p>
      <p className="mt-2 text-[15px] leading-relaxed">{gradeText}</p>
      {t.grade && (
        <p className="mt-1 text-[13px] text-muted">등급은 그 달을 학습하지 않은 모델이 매긴 점수로 정했어요.</p>
      )}
      {t.whipsaw && (
        <p className="mt-2 text-[14px] font-medium leading-relaxed text-touch">
          직전 5분 안에 반대 방향 돌파가 있었어요(휩쏘). 과거 분석에서 이런 재교차는 청산 신호로 쓸 만했어요.
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- outcome
function Locked({ title, until, note }: { title: string; until: string; note?: string }) {
  return (
    <div className="rounded-2xl border border-dashed border-line p-4">
      <p className="text-[13px] font-semibold text-muted">{title}</p>
      <div className="mt-1.5 flex items-start gap-2.5">
        <span aria-hidden="true" className="text-[17px]">
          🔒
        </span>
        <p className="text-[14px] leading-relaxed text-muted">
          재생이 <b className="tabular text-ink">{until}</b>을 지나면 열려요.
          {note ?? " 시스템도 이 시점에는 미래를 모릅니다."}
        </p>
      </div>
    </div>
  );
}

function Range({ w }: { w: Window }) {
  const lo = w.lo ?? 0;
  const hi = w.hi ?? 0;
  const m = Math.max(Math.abs(lo), Math.abs(hi), 0.02);
  return (
    <div>
      <div className="relative h-3 overflow-hidden rounded-full bg-sunk">
        <div className="absolute inset-y-0 left-1/2 w-px bg-line" />
        <div
          className="absolute inset-y-0 right-1/2 rounded-l-full bg-down"
          style={{ width: `${(Math.abs(Math.min(lo, 0)) / m) * 50}%` }}
        />
        <div
          className="absolute inset-y-0 left-1/2 rounded-r-full bg-up"
          style={{ width: `${(Math.max(hi, 0) / m) * 50}%` }}
        />
      </div>
      <div className="mt-1.5 flex justify-between text-[13px]">
        <span className="tabular font-semibold text-down">최저 {fmtPct(w.lo)}</span>
        <span className="tabular font-semibold text-up">최고 {fmtPct(w.hi)}</span>
      </div>
    </div>
  );
}

function Outcome({ e, idx }: { e: PEvent; idx: number }) {
  const o = e.outcome;
  if (!o) {
    return (
      <p className="rounded-2xl border border-dashed border-line p-4 text-[14px] text-muted">
        장 마감이 가까워 이후 결과를 잴 수 없는 알림이에요.
      </p>
    );
  }
  if (e.kind === "touch") {
    const t = o.touch!;
    if (idx < (e.revealAt.touch ?? 0))
      return <Locked title="모델이 맞히려던 결과 · 그 뒤 10분" until={t.reveal} />;
    const up = e.touch!.direction > 0;
    return (
      <div className="rounded-2xl border border-line p-4">
        <p className="text-[13px] font-semibold text-muted">모델이 맞히려던 결과 · 그 뒤 10분</p>
        <p className={`mt-1.5 text-[18px] font-bold ${t.hit ? "text-touch" : "text-muted"}`}>
          {t.hit ? `${up ? "+" : "−"}0.5% 터치 ✓` : `${up ? "+" : "−"}0.5%에 닿지 못함`}
        </p>
        <p className="mt-1 text-[14px] text-muted">
          그 사이 가장 불리했던 움직임 <span className="tabular font-semibold text-ink">{fmtPct(t.mae, 2)}</span>
        </p>
      </div>
    );
  }
  const h60 = o.h60;
  const eod = o.eod;
  const eodOpen = eod && idx >= (e.revealAt.eod ?? Infinity);
  return (
    <div className="space-y-3">
      {eod &&
        (eodOpen ? (
          <div className="rounded-2xl border border-line p-4">
            <p className="mb-2.5 text-[13px] font-semibold text-muted">모델이 맞히려던 결과 · 장 마감까지</p>
            <Range w={eod} />
            <div className="mt-3 space-y-2">
              {e.alerts!.map((a, i) => (
                <Verdict key={i} a={a} w={eod} />
              ))}
            </div>
          </div>
        ) : (
          <Locked
            title="모델이 맞히려던 결과 · 장 마감까지"
            until={eod.reveal}
            note=" 장 마감까지 지켜봐야 맞았는지 알 수 있어요."
          />
        ))}
      {h60 &&
        (idx < (e.revealAt.h60 ?? 0) ? (
          <Locked title={`참고 · 그 뒤 ${h60.full ? "60분" : "장 마감까지"} 움직임`} until={h60.reveal} />
        ) : (
          <div className="rounded-2xl border border-line p-4">
            <p className="mb-2.5 text-[13px] font-semibold text-muted">
              참고 · 그 뒤 {h60.full ? "60분" : "장 마감까지"} 움직임 (알림 가격 기준)
            </p>
            <Range w={h60} />
          </div>
        ))}
    </div>
  );
}

/** did the risk the model scored actually happen by the close? */
function Verdict({ a, w }: { a: Alert; w: Window }) {
  const long = a.side === "long";
  const happened = long ? (w.lo ?? 0) <= -0.015 : (w.hi ?? 0) >= 0.015;
  const what = long ? "−1.5% 이상 밀림" : "+1.5% 이상 튀어 오름";
  const band = BAND_KO[a.band] ?? a.band;
  return (
    <div className="flex items-center justify-between gap-3 rounded-xl bg-sunk px-3 py-2.5 text-[14px]">
      <span className="min-w-0">
        <span className="text-muted">{long ? "모델 위험" : "모델 반등위험"} </span>
        <b
          className={
            a.band === "HIGH" ? "text-risk-high" : a.band === "LOW" ? "text-risk-low" : "text-ink"
          }
        >
          {band}
        </b>
      </span>
      <span className={`shrink-0 font-semibold ${happened ? "text-ink" : "text-muted"}`}>
        실제: {happened ? `${what} ✓` : `${what} 없음`}
      </span>
    </div>
  );
}
