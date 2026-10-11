import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BuildStaleBanner } from "./BuildStaleBanner";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("BuildStaleBanner", () => {
  it("shows a banner when build-info version differs", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve({
          json: () => Promise.resolve({ version: "oldhash" }),
        }),
      ),
    );
    render(<BuildStaleBanner currentVersion="newhash" />);
    await waitFor(() => {
      expect(
        screen.getByText("界面版本过旧（浏览器缓存了老页面），点击刷新"),
      ).toBeInTheDocument();
    });
  });

  it("stays hidden when versions match", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve({
          json: () => Promise.resolve({ version: "same" }),
        }),
      ),
    );
    const { container } = render(<BuildStaleBanner currentVersion="same" />);
    await waitFor(() => {
      expect(fetch).toHaveBeenCalled();
    });
    expect(container.textContent).not.toContain("界面版本过旧");
  });
});
