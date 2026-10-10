import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

const ROW_H = 64;

interface Props<T> {
  items: T[];
  rowHeight?: number;
  overscan?: number;
  followTail: boolean;
  onFollowChange: (follow: boolean) => void;
  renderRow: (item: T, index: number) => ReactNode;
  getKey: (item: T, index: number) => string;
  scrollToKey?: string | null;
}

export function VirtualList<T>({
  items,
  rowHeight = ROW_H,
  overscan = 8,
  followTail,
  onFollowChange,
  renderRow,
  getKey,
  scrollToKey,
}: Props<T>) {
  const scrollerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [height, setHeight] = useState(400);

  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setHeight(el.clientHeight));
    ro.observe(el);
    setHeight(el.clientHeight);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!followTail) return;
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [items.length, followTail]);

  useEffect(() => {
    if (!scrollToKey) return;
    const idx = items.findIndex((it, i) => getKey(it, i) === scrollToKey);
    if (idx < 0) return;
    const el = scrollerRef.current;
    if (!el) return;
    el.scrollTop = Math.max(0, idx * rowHeight - height / 3);
    onFollowChange(false);
  }, [scrollToKey, items, getKey, rowHeight, height, onFollowChange]);

  const total = items.length * rowHeight;
  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const end = Math.min(
    items.length,
    Math.ceil((scrollTop + height) / rowHeight) + overscan,
  );
  const slice = useMemo(() => items.slice(start, end), [items, start, end]);

  return (
    <div
      ref={scrollerRef}
      className="traj-list"
      onScroll={() => {
        const el = scrollerRef.current;
        if (!el) return;
        setScrollTop(el.scrollTop);
        const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
        if (atBottom && !followTail) onFollowChange(true);
        if (!atBottom && followTail) onFollowChange(false);
      }}
    >
      <div className="virt-spacer" style={{ height: total }}>
        {slice.map((item, i) => {
          const index = start + i;
          return (
            <div
              key={getKey(item, index)}
              className="virt-row"
              style={{ top: index * rowHeight, height: rowHeight }}
            >
              {renderRow(item, index)}
            </div>
          );
        })}
      </div>
    </div>
  );
}
