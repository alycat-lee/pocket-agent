import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ProvidersCard } from "./ProvidersCard";
import { updateProvider } from "../api/client";

vi.mock("../api/client", () => ({
  fetchProviders: () =>
    Promise.resolve({
      providers: [
        {
          id: "p1",
          name: "本地模型",
          type: "llamacpp",
          base_url: "http://localhost:8000/v1",
          api_key: "",
          model: "local-model",
          sampling: null,
        },
      ],
      default_provider_id: "p1",
      routing: { choose_fill: "p1", generate: "p1" },
    }),
  fetchProviderPresets: () => Promise.resolve({ presets: [] }),
  testProvider: () =>
    Promise.resolve({
      ok: true,
      connect_ms: 1,
      generate_ms: 2,
      model_list: ["qwen3-8b-q4"],
      error: "",
    }),
  updateProvider: vi.fn(() => Promise.resolve({ ok: true, providers: [] })),
  createProvider: vi.fn(),
  deleteProvider: vi.fn(),
  setDefaultProvider: vi.fn(),
  updateRouting: vi.fn(),
}));

describe("ProvidersCard", () => {
  it("backfills model from test result and persists it", async () => {
    render(<ProvidersCard />);
    await waitFor(() => {
      expect(screen.getByDisplayValue("local-model")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByRole("button", { name: "测试" }));
    await waitFor(() => {
      expect(screen.getByDisplayValue("qwen3-8b-q4")).toBeInTheDocument();
    });
    expect(updateProvider).toHaveBeenCalledWith(
      "p1",
      expect.objectContaining({ model: "qwen3-8b-q4" }),
    );
  });
});
