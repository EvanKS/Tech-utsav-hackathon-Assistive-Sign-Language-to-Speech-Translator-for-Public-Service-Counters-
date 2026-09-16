import sys
sys.path.insert(0, '.')
from ml.temporal_smoothing import SignStabilizer, UNKNOWN

s = SignStabilizer(window=5, min_conf=0.7, min_agreement=0.6, cooldown_ms=100, exit_frames=3)

print("=== HELLO phase ===")
for i in range(6):
    r = s.push("hello", 0.9, timestamp_ms=i*100)

print(f"After hello: last_emitted={s._last_emitted}")

print("\n=== GAP phase ===")
for i in range(6):
    r = s.push("none", 0.3, timestamp_ms=700+i*100)

print(f"After gap: held={s._held_label}, last_emitted={s._last_emitted}")

print("\n=== YES phase - detailed ===")
for i in range(8):
    ts = 2000+i*100
    r = s.push("yes", 0.9, timestamp_ms=ts)
    d = r["debug"]
    time_since = ts - s._last_emit_time
    in_cool = time_since < s.cooldown_ms
    
    modal = d["modal_label"]
    agree = d["agreement"]
    
    check1 = modal != UNKNOWN
    check2 = agree >= s.min_agreement
    check3 = not in_cool
    check4 = modal != s._last_emitted
    check5 = s._held_label is None or modal != s._held_label
    
    print(f"Yes {i}: modal={modal}, agree={agree:.2f}, time_since={time_since}ms")
    print(f"  checks: not_unknown={check1}, agree_ok={check2}, not_cooldown={check3}, diff_last={check4}, held_check={check5}")
    print(f"  result: emitted={r['emitted']}, held={s._held_label}")
