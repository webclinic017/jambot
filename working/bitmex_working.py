
# %% - IMPORTS

if True:
    SYMBOL = 'XBTUSD'
    import json

    from jambot import functions as f
    from jambot.exchanges.bitmex import Bitmex
    from jambot.tradesys import orders as ords
    from jambot.tradesys.enums import OrderStatus, OrderType
    from jambot.tradesys.orders import ExchOrder, LimitOrder


# %% - INIT EXCHANGE
exch = Bitmex.default(test=True, refresh=True)

order = ExchOrder.from_dict(exch.orders[0])
