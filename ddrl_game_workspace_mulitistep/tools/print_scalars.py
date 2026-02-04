from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os, glob, sys

logdir = 'logs'
files = glob.glob(os.path.join(logdir, 'events.out.tfevents.*'))
if not files:
    print('No event files found in', logdir)
    sys.exit(0)
files.sort(key=os.path.getmtime)
latest = files[-1]
print('Using event file:', latest)

ea = EventAccumulator(latest)
ea.Reload()

wanted = ['Train/loss', 'Train/epsilon', 'Train/memory_size', 'Reward/avg10']
scalars = ea.Tags().get('scalars', [])
for tag in wanted:
    if tag in scalars:
        events = ea.Scalars(tag)
        print(f"--- {tag} (last {min(10, len(events))}) ---")
        for e in events[-10:]:
            print(f"step={e.step}, wall_time={e.wall_time:.0f}, value={e.value}")
    else:
        print(f"Tag not found in events: {tag}")
