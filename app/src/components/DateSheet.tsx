import { useRef, useState } from "react";
import { Drawer } from "vaul";
import type { AvailableDate } from "../types";
import { dateKo } from "../lib/prepare";

const WEEKDAYS = ["일", "월", "화", "수", "목", "금", "토"];

export default function DateSheet({
  dates, selected, onSelect, onClose,
}: {
  dates: AvailableDate[];
  selected: string | null;
  onSelect: (date: string) => void;
  onClose: () => void;
}) {
  const returnFocus = useRef(document.activeElement as HTMLElement | null);
  const available = new Set(dates.map((d) => d.date));
  const sorted = [...available].sort();
  const months = [...new Set(sorted.map((d) => d.slice(0, 7)))];
  const [month, setMonth] = useState(
    selected && available.has(selected) ? selected.slice(0, 7) : months[months.length - 1],
  );
  const monthIndex = months.indexOf(month);
  const [year, monthNumber] = month.split("-").map(Number);
  const offset = new Date(year, monthNumber - 1, 1).getDay();
  const days = new Date(year, monthNumber, 0).getDate();

  return (
    <Drawer.Root open onOpenChange={(open) => !open && onClose()}>
      <Drawer.Portal>
        <Drawer.Overlay className="fixed inset-0 z-40 bg-black/45" />
        <Drawer.Content
          className="fixed inset-x-0 bottom-0 z-50 mx-auto flex max-h-[92dvh] max-w-md flex-col rounded-t-3xl bg-surface text-ink outline-none"
          aria-describedby="date-description"
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            returnFocus.current?.focus();
          }}
        >
          <div className="mx-auto mt-2.5 h-1.5 w-10 shrink-0 rounded-full bg-line" />
          <div className="overflow-y-auto px-4 pt-3" style={{ paddingBottom: "max(env(safe-area-inset-bottom), 20px)" }}>
            <div className="flex items-center justify-between">
              <Drawer.Title className="text-[21px] font-bold">날짜 선택</Drawer.Title>
              <button onClick={onClose} className="h-11 rounded-xl px-3 text-[14px] font-semibold text-muted">
                닫기
              </button>
            </div>
            <Drawer.Description id="date-description" className="mt-1 text-[13px] leading-relaxed text-muted">
              {sorted[0].replaceAll("-", ".")} – {sorted[sorted.length - 1].replaceAll("-", ".")}
              <br />검증한 {sorted.length}거래일 · 날짜를 누르면 바로 재생해요.
            </Drawer.Description>
            <div className="mt-4 flex items-center justify-between">
              <button
                aria-label="이전 달"
                disabled={monthIndex <= 0}
                onClick={() => setMonth(months[monthIndex - 1])}
                className="h-11 w-11 rounded-xl bg-sunk text-xl disabled:cursor-default disabled:opacity-25"
              >‹</button>
              <p aria-live="polite" className="tabular text-[17px] font-bold">{year}년 {monthNumber}월</p>
              <button
                aria-label="다음 달"
                disabled={monthIndex >= months.length - 1}
                onClick={() => setMonth(months[monthIndex + 1])}
                className="h-11 w-11 rounded-xl bg-sunk text-xl disabled:cursor-default disabled:opacity-25"
              >›</button>
            </div>
            <div className="mt-3 grid grid-cols-7 gap-1 text-center">
              {WEEKDAYS.map((day) => <span key={day} className="py-2 text-[12px] text-muted">{day}</span>)}
              {Array.from({ length: offset }, (_, i) => <span key={`empty-${i}`} />)}
              {Array.from({ length: days }, (_, i) => {
                const day = i + 1;
                const date = `${month}-${String(day).padStart(2, "0")}`;
                const enabled = available.has(date);
                const current = date === selected;
                return (
                  <button
                    key={date}
                    disabled={!enabled}
                    aria-label={`${dateKo(date)}${enabled ? "" : " · 데이터 없음"}`}
                    aria-pressed={current}
                    onClick={() => onSelect(date)}
                    className={`tabular h-11 rounded-xl text-[15px] font-semibold transition ${
                      current ? "bg-ink text-surface" : enabled ? "bg-sunk hover:bg-line" : "cursor-default text-muted opacity-30"
                    }`}
                  >{day}</button>
                );
              })}
            </div>
            <p className="mt-4 text-center text-[12px] text-muted">흐린 날짜는 데이터가 없어요.</p>
          </div>
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}
