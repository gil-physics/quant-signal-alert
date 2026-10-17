let openedFromHome = false;

/** Home -> replay: a real history entry, so the phone's back gesture returns home. */
export function openDate(date: string) {
  openedFromHome = true;
  window.location.hash = `#/d/${date}`;
}

export function goHome() {
  if (openedFromHome) {
    openedFromHome = false;
    window.history.back();
  } else {
    window.location.hash = "#/";
  }
}

/** "#/d/2026-06-19" -> "2026-06-19"; anything else -> null (home) */
export function dateFromHash(): string | null {
  const m = window.location.hash.match(/^#\/d\/(\d{4}-\d{2}-\d{2})$/);
  if (!m) openedFromHome = false;
  return m ? m[1] : null;
}
