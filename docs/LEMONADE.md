# PolyStrata in Lemonade

[Lemonade](https://github.com/lemonade-sdk/lemonade) can use a running PolyStrata server as a provider. Its model
then shows in Lemonade's model list as `polystrata.<model>` and works in every app that talks to Lemonade.
Lemonade itself is not modified.

## Steps

1. Start PolyStrata as usual (`serve/server.py --engine strata --config <your config> --port 8095`) and wait for
   `ready: http://127.0.0.1:8095/v1`.
2. Register it with Lemonade. Lemonade only lists a provider's models once it has an API key for it; PolyStrata
   needs none unless your config sets `"api_key"`, so any value works. Over plain HTTP, Lemonade also asks you to
   allow that explicitly:

   ```sh
   lemonade cloud install polystrata --base-url http://127.0.0.1:8095/v1 --allow-insecure-http
   lemonade cloud auth polystrata --api-key <your config's api_key, or any text> --allow-insecure-http
   lemonade cloud list        # polystrata ... models_discovered=1
   ```

   Add `--host`/`--port` to `lemonade` when your Lemonade server is not on 127.0.0.1:13305.
3. Chat through Lemonade with the model id `polystrata.<model_name from your config>`.

Give thinking models room to answer: a reply that reaches `max_tokens` while still thinking has no answer
(`content` is `""`).

## Tested

2026-10-30, Lemonade 2026.43.3 on a second machine, PolyStrata v0.1.43.2-sm60-compat with
Qwen3.6-35B-A3B Q5_K_M on one Tesla P100 16 GB (2562 of 10496 routed experts in VRAM, the rest on an AVX-only
Xeon), context 16384:

| Request | Through | Tokens | Speed |
| --- | --- | --- | --- |
| first answer after start | PolyStrata directly | 216 | 14.2 tok/s |
| later answer | Lemonade (`polystrata.qwen3.6-35b-a3b`) | 243 | 47.7 tok/s |

The first answer is slower while the expert cache fills (73% hits later).
