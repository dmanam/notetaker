{
  description = "Math lecture video to typeset notes";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSystem = f: nixpkgs.lib.genAttrs systems f;

      mkShell = system: cuda:
        let
          # allowUnfree for the claude/codex CLIs (and CUDA when enabled).
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
              cudaSupport = cuda;
            };
          };

          pythonEnv = pkgs.python3.withPackages (ps: with ps; [
            openai-whisper
            yt-dlp
            tqdm
            click
            anthropic
            claude-agent-sdk   # subscription backend (drives the claude CLI)
            mcp                # serves our tools to the codex backend
            modal              # remote GPU transcription
            pymupdf            # PDF text extraction (better than pypdf)
            pypdf              # fallback extractor
            requests
          ]);
        in pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.ffmpeg
            pkgs.yt-dlp
            pkgs.claude-code   # `claude` CLI for the subscription backend
            pkgs.codex         # `codex` CLI for the codex backend
          ] ++ pkgs.lib.optionals cuda [
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
          ];

          shellHook = ''
            echo "notetaker dev shell ready${pkgs.lib.optionalString cuda " (CUDA)"}"
            echo "  python: $(python3 --version)"
            echo "  yt-dlp: $(yt-dlp --version)"
            echo "  ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
          '' + pkgs.lib.optionalString cuda ''
            export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
            export LD_LIBRARY_PATH="${
              pkgs.lib.makeLibraryPath [
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.cudnn
              ]
            }:$LD_LIBRARY_PATH"
            echo "  CUDA: ${pkgs.cudaPackages.cudatoolkit.version} (GPU visibility requires host drivers)"
          '';
        };
    in {
      devShells = forEachSystem (system:
        {
          # CUDA is off by default — transcription runs on Modal
          # (--transcribe modal). Use `nix develop .#cuda` for local GPU
          # Whisper (x86_64-linux only).
          default = mkShell system false;
        } // nixpkgs.lib.optionalAttrs (system == "x86_64-linux") {
          cuda = mkShell system true;
        });
    };
}
