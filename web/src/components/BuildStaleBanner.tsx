import { useEffect, useState } from "react";

interface Props {
  currentVersion?: string;
  infoUrl?: string;
}

export function BuildStaleBanner({
  currentVersion = __BUILD_VERSION__,
  infoUrl = "/static/build-info.json",
}: Props) {
  const [stale, setStale] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void fetch(infoUrl, { cache: "no-store" })
      .then((res) => res.json())
      .then((data: { version?: string }) => {
        if (cancelled) return;
        if (typeof data.version === "string" && data.version !== currentVersion) {
          setStale(true);
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [currentVersion, infoUrl]);

  if (!stale) return null;
  return (
    <button
      type="button"
      className="build-stale-banner"
      onClick={() => {
        window.location.reload();
      }}
    >
      界面版本过旧（浏览器缓存了老页面），点击刷新
    </button>
  );
}
