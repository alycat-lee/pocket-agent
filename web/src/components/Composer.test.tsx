import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Composer } from "./Composer";

describe("Composer", () => {
  it("has workspace and permission selectors only — no intent badge", () => {
    render(
      <Composer
        permission="workspace"
        workspaces={[{ id: "w1", name: "默认", root: "." }]}
        workspaceId="w1"
        value=""
        running={false}
        promptCharBudget={5200}
        segments={[]}
        onPermission={vi.fn()}
        onWorkspace={vi.fn()}
        onChange={vi.fn()}
        onSend={vi.fn()}
        onStop={vi.fn()}
      />,
    );
    expect(screen.getByLabelText("工作区")).toBeInTheDocument();
    expect(screen.getByLabelText("权限")).toBeInTheDocument();
    expect(screen.queryByText(/自动识别/)).not.toBeInTheDocument();
    expect(screen.queryByText(/点击纠正/)).not.toBeInTheDocument();
    expect(document.querySelector(".intent-badge")).toBeNull();
  });
});
