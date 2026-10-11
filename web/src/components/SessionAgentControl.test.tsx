import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { AgentInfo } from "../api/types";
import { SessionAgentControl } from "./SessionAgentControl";

function agent(name: string): AgentInfo {
  return {
    name,
    description: "",
    provider_id: null,
    tools: null,
    permission: "workspace",
    prompt: "",
    root: `agents/${name}`,
  };
}

describe("SessionAgentControl", () => {
  it("renders a static badge without select for a single agent", () => {
    render(
      <SessionAgentControl
        agents={[agent("default")]}
        agentName="default"
        onSelectAgent={vi.fn()}
      />,
    );
    expect(screen.getByText("default")).toBeInTheDocument();
    expect(screen.queryByRole("combobox")).toBeNull();
  });

  it("renders a select for multiple agents and notifies on change", () => {
    const onSelect = vi.fn();
    render(
      <SessionAgentControl
        agents={[agent("default"), agent("写手")]}
        agentName="default"
        onSelectAgent={onSelect}
      />,
    );
    const select = screen.getByRole("combobox");
    fireEvent.change(select, { target: { value: "写手" } });
    expect(onSelect).toHaveBeenCalledWith("写手");
  });
});
