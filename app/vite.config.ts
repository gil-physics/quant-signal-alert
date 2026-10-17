import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["icon.svg", "apple-touch-icon.png"],
      manifest: {
        name: "quant 리플레이",
        short_name: "리플레이",
        description: "과거 하루 장을 1분씩 다시 재생하며 알림이 뜨는 과정을 보는 앱",
        lang: "ko",
        start_url: "/",
        display: "standalone",
        orientation: "portrait",
        background_color: "#0e1116",
        theme_color: "#0e1116",
        icons: [
          { src: "icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "icon-512.png", sizes: "512x512", type: "image/png" },
          { src: "icon-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
      },
      workbox: {
        globPatterns: ["**/*.{js,css,html,svg,png}"],
        // replay days are fetched on demand; once played they stay available offline
        runtimeCaching: [
          {
            // Korean font subsets load on demand (only the glyphs a page uses)
            urlPattern: ({ request }) => request.destination === "font",
            handler: "CacheFirst",
            options: { cacheName: "fonts", expiration: { maxEntries: 120 } },
          },
          {
            urlPattern: ({ url }) => url.pathname.startsWith("/data/"),
            handler: "StaleWhileRevalidate",
            options: { cacheName: "replay-data", expiration: { maxEntries: 60 } },
          },
        ],
      },
    }),
  ],
  build: { target: "es2022", sourcemap: false },
});
