local wezterm = require 'wezterm'
return {
  enable_tab_bar = false,
  check_for_updates = false,
  ssh_domains = {},
  initial_cols = 140,
  initial_rows = 42,
  font_size = 11,
  term = 'xterm-256color',
  exit_behavior = 'CloseOnCleanExit',
  window_close_confirmation = 'AlwaysPrompt',
  disable_default_key_bindings = true,
  keys = {
    { key = 'C', mods = 'CTRL|SHIFT', action = wezterm.action.CopyTo 'Clipboard' },
    { key = 'V', mods = 'CTRL|SHIFT', action = wezterm.action.PasteFrom 'Clipboard' },
    { key = '+', mods = 'CTRL|SHIFT', action = wezterm.action.IncreaseFontSize },
    { key = '_', mods = 'CTRL|SHIFT', action = wezterm.action.DecreaseFontSize },
  },
}
