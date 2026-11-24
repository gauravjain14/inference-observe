# Keep the live stack running

A user timer runs `scripts/live.sh ensure` one minute after boot and every two minutes after that. It restarts the
whole stack if the engine, the proxy or the collector has stopped, and otherwise restarts the traffic and the push
when they have stopped. Settings come from `scripts/live.env`.

```bash
mkdir -p ~/.config/systemd/user
sed "s#%h/inference-observe#$PWD#" scripts/systemd/inference-observe.service > ~/.config/systemd/user/inference-observe.service
cp scripts/systemd/inference-observe.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now inference-observe.timer
sudo loginctl enable-linger "$USER"   # run at boot, before anyone logs in
```

- `scripts/live.sh down` stops the stack, and the timer leaves it stopped until `scripts/live.sh up`.
- `scripts/live.sh offline` stops only the push, and the timer leaves it stopped until `scripts/live.sh push`.
- Check the timer with `systemctl --user list-timers inference-observe.timer` and `journalctl --user -u inference-observe`.
- Remove it with `systemctl --user disable --now inference-observe.timer`.
