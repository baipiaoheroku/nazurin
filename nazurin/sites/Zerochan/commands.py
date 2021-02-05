from aiogram.dispatcher import filters
from aiogram.types import Message

from nazurin import bot
from nazurin.utils.exceptions import NazurinError

from .api import Zerochan

api = Zerochan()

@bot.handler(filters.RegexpCommandsFilter(regexp_commands=[r'/zerochan (\S+)'])
             )
async def zerochan_view(message: Message, regexp_command):
    try:
        post_id = int(regexp_command.group(1))
        if post_id < 0:
            await message.reply('Invalid post id!')
            return
        imgs, caption = await api.view(post_id)
        await bot.sendPhotos(message, imgs, caption)
    except (IndexError, ValueError):
        await message.reply('Usage: /zerochan <post_id>')
    except NazurinError as error:
        await message.reply(error.msg)

@bot.handler(
    filters.RegexpCommandsFilter(regexp_commands=[r'/zerochan_download (\S+)'])
)
async def zerochan_download(message: Message, regexp_command):
    try:
        post_id = int(regexp_command.group(1))
        if post_id <= 0:
            await message.reply('Invalid post id!')
            return
        imgs = await api.download(post_id)
        await bot.sendDocuments(message, imgs)
    except (IndexError, ValueError):
        await message.reply('Usage: /zerochan_download <post_id>')
    except NazurinError as error:
        await message.reply(error.msg)

commands = [
    zerochan_view,
    zerochan_download,
]