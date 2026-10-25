import { Drawer } from "vaul";

const ITEMS = [
  {
    icon: "↑↓",
    tone: "text-up",
    title: "롱 엔진 (▲)",
    body: "규칙이 방향 알림(상방 가능성, 하방 위험 등)을 내고, LightGBM 모델이 그 순간부터 장 마감까지 −1.5% 이상 밀릴 위험 점수를 붙입니다.",
  },
  {
    icon: "↓",
    tone: "text-down",
    title: "하방 시스템 (🔻)",
    body: "롱 엔진과 대칭인 하방 알림입니다. 모델은 반대로 +1.5% 이상 튀어 오를 위험 점수를 붙입니다.",
  },
  {
    icon: "✦",
    tone: "text-touch",
    title: "이평 돌파",
    body: "5분선이 20분선을 넘는 순간을 알리고, CNN 모델이 10분 안에 0.5% 더 갈 가능성을 A–D 등급으로 매깁니다.",
  },
];

export default function AboutSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <Drawer.Root open={open} onOpenChange={(o) => !o && onClose()}>
      <Drawer.Portal>
        <Drawer.Overlay className="fixed inset-0 z-40 bg-black/45" />
        <Drawer.Content
          className="fixed inset-x-0 bottom-0 z-50 mx-auto flex max-h-[88dvh] max-w-xl flex-col rounded-t-3xl bg-surface text-ink outline-none"
          aria-describedby={undefined}
        >
          <div className="mx-auto mt-2.5 h-1.5 w-10 shrink-0 rounded-full bg-line" />
          <div
            className="min-h-0 overflow-y-auto px-5 pt-3"
            style={{ paddingBottom: "max(env(safe-area-inset-bottom), 20px)" }}
          >
            <Drawer.Title className="text-[21px] font-bold">이 앱이 보여 주는 것</Drawer.Title>
            <p className="mt-1.5 text-[15px] leading-relaxed text-muted">
              실제 알림 프로그램을 그날 그대로 다시 돌린 결과를 1분씩 보여 줍니다. 알림은 세 종류입니다.
            </p>
            <ul className="mt-4 space-y-3">
              {ITEMS.map((it) => (
                <li key={it.title} className="flex gap-3 rounded-2xl bg-sunk p-4">
                  <span className={`w-7 shrink-0 text-center text-[18px] font-bold ${it.tone}`}>{it.icon}</span>
                  <div>
                    <p className="text-[16px] font-semibold">{it.title}</p>
                    <p className="mt-1 text-[14px] leading-relaxed text-muted">{it.body}</p>
                  </div>
                </li>
              ))}
            </ul>
            <ul className="mt-4 space-y-2 text-[14px] leading-relaxed">
              <li>• 색은 국내 관례대로 빨강이 상승, 파랑이 하락입니다.</li>
              <li>• 모든 날짜는 모델이 학습하지 않은 기간이므로 성적이 부풀려지지 않습니다.</li>
              <li>• 알림의 결과는 그 시간이 지나야 열립니다. 시스템도 그 순간에는 미래를 모르기 때문입니다.</li>
              <li>• 교육용 리플레이입니다. 주문을 내지 않으며 투자 권유가 아닙니다.</li>
            </ul>
            <button
              onClick={onClose}
              className="mt-5 h-12 w-full rounded-2xl bg-ink text-[16px] font-semibold text-surface"
            >
              닫기
            </button>
          </div>
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}
