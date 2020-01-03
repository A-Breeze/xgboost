"""An exact copy of config_private.py, but with the private values replaced by None so it can be committed to Git

If you need to use any of these variables, take a copy of this script and call it ``config_private.py``.
It will be ignored by Git, so you can replace the ``None`` values with your private variables.
"""
# If you need to access the internet via a proxy, set it here
# (the same address for both items works for me)
proxy_dict = {"http": None, "https": None}
