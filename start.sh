#!/bin/bash
source .env
cloudflared tunnel run --token $CLOUDFLARED_TOKEN & # run this in the background using &
CLOUDFLARED_PID=$! # get the PID
trap 'echo "FastAPI exited." && kill $CLOUDFLARED_PID 2> /dev/null' EXIT # kill the process now
fastapi dev
# when I press CTRL-C (SIGINT), it's passed to all child processes as well
# so cloudflared exits early, and the script will say "no such process" for the trap.
# But I have written "overwrite to /dev/null (the blackhole) on descriptor 2 (stderr)", so the error won't clutter my screen.
