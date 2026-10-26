from prisma import Prisma

db = Prisma(auto_register=True)


async def connect_db() -> None:
    """Connect to the MySQL database via Prisma."""
    if not db.is_connected():
        await db.connect()


async def disconnect_db() -> None:
    """Disconnect from the MySQL database."""
    if db.is_connected():
        await db.disconnect()
