******************************************
Federated Learning Client Package: GMIC-FL
******************************************

This package contains the necessary components to participate in a secure federated learning (FL) job using a mammography breast cancer detection model.

---------------------
What's Included
---------------------
startup/
  ├── start.sh              - Launches the FL client
  ├── sub_start.sh          - Internal launch script
  ├── stop_fl.sh            - Stops the FL client
  ├── client.crt            - Your client certificate
  ├── client.key            - Your private key (KEEP SAFE)
  ├── rootCA.pem            - Root certificate authority
  ├── fed_client.json       - Client configuration file
  └── signature.json        - Signature metadata (do not edit)

local/                             - Site-local policy/config templates for this org (see section below)
   ├── authorization.json.default  - Template for admin authorization rules (who can run which admin commands against this client).
   ├── log_config.json.default     - Template for Python logging (levels/handlers/rotation for client logs).
   ├── privacy.json.sample         - Example privacy / data-filter pipeline (e.g., de-identification); copy to privacy.json to enable.
   └── resources.json.default      - Template for local resource hints (e.g., GPU/CPU/concurrency limits) used by the client runtime.

readme.txt                  - This file
Dockerfile                  - Image definition
docker-compose.yml          - Defines container behavior
run_client.sh               - Linux/macOS entry point to build and run
run_client_windows.ps1      - Windows PowerShell entry point to build and run

Other directories:
  transfer/                 - NVFLARE job payloads (if provided)
  data/                     - Your data directory (mounted at runtime; not required to live inside this package)
  models/                   - Starting model weights (optional)

------------------------------------------
Org-Specific Local Configuration (org/local)
------------------------------------------
The org folder (e.g., <your-site>/) contains a `local/` directory with **site-local configuration templates**. These files control behavior on the client host only (logging, authorization, privacy filters, resource hints). They do **not** change the federated training protocol itself.

How to use:
- Files ending in `.default` or `.sample` are **templates**. To activate a customization, copy/rename the template to the same name **without** the suffix and edit it:
  - `authorization.json.default`  → `authorization.json`
  - `log_config.json.default`     → `log_config.json`
  - `privacy.json.sample`         → `privacy.json`
  - `resources.json.default`      → `resources.json`
- If an unsuffixed file exists, the client uses it; otherwise, the client falls back to built-in defaults.

What each file controls:
- **authorization.json**  
  Defines which admin users/roles may execute specific admin commands on this client (least-privilege recommended).
- **log_config.json**  
  Python logging config (levels, formatters, handlers, rotation). Increase verbosity during debugging; revert to INFO for production.
- **privacy.json**  
  Optional privacy/data-filter pipeline (e.g., de-identification transforms) applied locally before any data/metadata leaves the client.
- **resources.json**  
  Local resource hints and limits (e.g., max concurrent tasks, GPU visibility/preferences). Adjust to match the machine’s capacity.

Notes:
- Keep these files in the delivered relative paths; if you move the org folder, ensure any references in `fed_client.json` remain valid.
- Treat `privacy.json` as sensitive if it documents PHI handling; store and share appropriately.

---------------------
Running the FL Client
---------------------

1. Prepare Your Data
   - Place your mammogram data in a local directory (e.g., ./my_data).
   - This will be mounted (read-only by default) to /workspace/data inside the container.

2. Run the Client
   Use the helper script to build and launch the client with your desired GPUs:

   Linux/macOS:
     ./scripts/run_client.sh /path/to/my_data [gpus]

   Windows (PowerShell):
     .\scripts\run_client_windows.ps1 -DataDir "C:\path\to\my_data" -GPUs "all|none|0,1"

   Examples (Linux/macOS):
     ./scripts/run_client.sh ./data           - Use all GPUs
     ./scripts/run_client.sh ./data 0,1       - Use specific GPUs
     ./scripts/run_client.sh ./data none      - Run with no GPU

3. Stop the Client
   To stop the client container:

   docker compose down

---------------------
Security Notes
---------------------

- Your identity in this FL project is embedded in the client.crt certificate (Common Name field).
- Your private key (client.key) must be kept secure and never shared.
- The certificates and fed_client.json must be kept together. If you move them, update the paths in fed_client.json accordingly.

---------------------
Advanced
---------------------

To inspect logs or access a shell in the running container:

   docker exec -it gmic_fl_container bash

To rebuild the container without launching:

   docker compose build

---------------------
Support
---------------------

If you encounter issues or have questions, please contact the study coordinator or technical lead.
