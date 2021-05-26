
# %% - IMPORTS

if True:
    SYMBOL = 'XBTUSD'
    import json

    from jambot import functions as f
    from jambot.exchanges.bitmex import Bitmex
    from jambot.tradesys import orders as ords
    from jambot.tradesys.enums import OrderStatus, OrderType
    from jambot.tradesys.orders import BitmexOrder, LimitOrder


# %% - INIT EXCHANGE
exch = Bitmex.default(test=True, refresh=True)

order = BitmexOrder.from_dict(exch.orders[0])
