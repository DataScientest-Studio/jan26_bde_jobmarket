from src.ingest.france_travail_client import FranceTravailClient
import time
from collections import deque


def main():
    client = FranceTravailClient()

    total_calls = 0
    t0 = time.time()
    prev_call_t = None

    # Fenêtre glissante 1 seconde pour un RPS "instantané"
    window_s = 1.0
    call_times = deque()

    n_calls_to_test = 30

    for i in range(n_calls_to_test):
        call_start = time.time()

        # Appel 1 par 1 via range
        client.get(
            "/partenaire/offresdemploi/v2/offres/search",
            params={"range": f"{i}-{i}"},
        )

        total_calls += 1
        now = time.time()

        # Δ entre deux appels
        delta = None
        if prev_call_t is not None:
            delta = call_start - prev_call_t
        prev_call_t = call_start

        # Mise à jour fenêtre glissante
        call_times.append(now)
        while call_times and (now - call_times[0] > window_s):
            call_times.popleft()

        # RPS instantané (fenêtre 1s)
        inst_rps = len(call_times) / window_s

        # RPS moyen global
        elapsed = now - t0
        avg_rps = total_calls / elapsed if elapsed > 0 else 0.0

        # Affichage
        if delta is None:
            print(
                f"[{total_calls:03d}] appel | avg={avg_rps:.2f} req/s | inst={inst_rps:.2f} req/s"
            )
        else:
            print(
                f"[{total_calls:03d}] appel | Δ={delta:.4f}s | avg={avg_rps:.2f} req/s | inst={inst_rps:.2f} req/s"
            )

    # Résumé final
    total_elapsed = time.time() - t0
    final_avg_rps = total_calls / total_elapsed if total_elapsed > 0 else 0.0
    print("\n--- Résumé ---")
    print(f"Nombre total d'appels : {total_calls}")
    print(f"Durée totale          : {total_elapsed:.2f}s")
    print(f"Moyenne               : {final_avg_rps:.2f} appels/s")


if __name__ == "__main__":
    main()
