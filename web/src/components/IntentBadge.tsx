import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { IntentInfo, IntentMode } from "../api/types";
import { badgeLabel, MODE_LABEL } from "../lib/intent";

interface Props {
  intent: IntentInfo;
  onCorrect: (mode: IntentMode) => void;
}

const OPTIONS: IntentMode[] = ["qa", "agent", "writer"];

export function IntentBadge({ intent, onCorrect }: Props) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const btnRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const menuId = useId();

  useEffect(() => {
    if (!open) return;
    const onDoc = (ev: MouseEvent) => {
      const t = ev.target as Node;
      if (btnRef.current?.contains(t) || menuRef.current?.contains(t)) return;
      setOpen(false);
    };
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        className={`intent-badge intent-${intent.mode}`}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={menuId}
        title="点击纠正本轮意图"
        onClick={() => {
          const rect = btnRef.current?.getBoundingClientRect();
          if (rect) {
            setPos({ top: rect.bottom + 6, left: rect.left });
          }
          setOpen((v) => !v);
        }}
      >
        {badgeLabel(intent)}
      </button>
      {open
        ? createPortal(
            <div
              ref={menuRef}
              id={menuId}
              className="intent-menu"
              role="menu"
              style={{ top: pos.top, left: pos.left }}
            >
              <div className="intent-menu-title">纠正为本轮意图</div>
              {OPTIONS.map((mode) => (
                <button
                  key={mode}
                  type="button"
                  role="menuitem"
                  className={mode === intent.mode ? "current" : ""}
                  disabled={mode === intent.mode}
                  onClick={() => {
                    setOpen(false);
                    onCorrect(mode);
                  }}
                >
                  {MODE_LABEL[mode]}
                </button>
              ))}
            </div>,
            document.body,
          )
        : null}
    </>
  );
}
