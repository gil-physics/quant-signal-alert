const GA_ID = import.meta.env.VITE_GA_MEASUREMENT_ID ?? "";
const LOCAL_HOST = /^(localhost|127\.0\.0\.1|\[::1\])$/;

declare global {
  interface Window {
    dataLayer: unknown[];
    gtag: (...args: unknown[]) => void;
  }
}

/** GA4 visitor stats. Off without a measurement id and on local dev hosts. */
export function initAnalytics() {
  if (!GA_ID || LOCAL_HOST.test(window.location.hostname)) return;

  window.dataLayer = window.dataLayer || [];
  window.gtag = function gtag() {
    // gtag.js reads the Arguments object itself, not a copied array
    // eslint-disable-next-line prefer-rest-params
    window.dataLayer.push(arguments);
  };
  window.gtag("js", new Date());
  window.gtag("config", GA_ID);

  const script = document.createElement("script");
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${encodeURIComponent(GA_ID)}`;
  document.head.appendChild(script);

  // which replay days people open: "#/d/2026-06-19" -> replay_date=2026-06-19
  let lastDate: string | null = null;
  const onRoute = () => {
    const m = window.location.hash.match(/^#\/d\/(\d{4}-\d{2}-\d{2})$/);
    const date = m ? m[1] : null;
    if (date && date !== lastDate) window.gtag("event", "open_replay_date", { replay_date: date });
    lastDate = date;
  };
  onRoute();
  window.addEventListener("hashchange", onRoute);
}
