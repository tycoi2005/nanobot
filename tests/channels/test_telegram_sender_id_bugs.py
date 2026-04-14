
import pytest
from unittest.mock import MagicMock, AsyncMock
from nanobot.channels.telegram import TelegramChannel, TelegramConfig
from nanobot.bus.queue import MessageBus

@pytest.fixture
def channel():
    config = TelegramConfig(enabled=True, token="123:abc")
    return TelegramChannel(config, MessageBus())

def test_sender_id_with_minimal_user(channel):
    """Reproduction: User with no username, no full_name, no first_name."""
    user = MagicMock()
    user.id = 12345
    user.username = None
    user.first_name = None
    user.full_name = None
    user.is_bot = False
    
    # It should at least return the ID
    sid = channel._sender_id(user)
    assert sid == "12345"

def test_sender_id_with_only_first_name(channel):
    """User with no username, but has first_name."""
    user = MagicMock()
    user.id = 67890
    user.username = None
    user.first_name = "John"
    user.full_name = "John" # PTB property behavior
    user.is_bot = False
    
    sid = channel._sender_id(user)
    assert sid == "67890|John"

def test_sender_id_with_full_name(channel):
    """User with no username, but has both names."""
    user = MagicMock()
    user.id = 11223
    user.username = None
    user.first_name = "John"
    user.full_name = "John Doe"
    user.is_bot = False
    
    sid = channel._sender_id(user)
    assert sid == "11223|John Doe"

def test_sender_id_is_bot(channel):
    """User is a bot."""
    user = MagicMock()
    user.id = 999
    user.username = "my_bot"
    user.full_name = "My Bot"
    user.is_bot = True
    
    sid = channel._sender_id(user)
    assert sid == "999|my_bot [BOT]"

@pytest.mark.asyncio
async def test_is_allowed_with_formatted_id(channel):
    """Ensure is_allowed handles the formatted ID correctly with allow_from."""
    channel.config.allow_from = ["12345"]
    
    # 1. Matching by ID in formatted string
    assert channel.is_allowed("12345|John Doe") is True
    
    # 2. Matching by username in formatted string
    channel.config.allow_from = ["john_username"]
    assert channel.is_allowed("12345|john_username") is True
    
    # 3. Denying when neither matches
    assert channel.is_allowed("67890|Other") is False
    
@pytest.mark.asyncio
async def test_on_message_ignores_self(channel):
    """Verify that messages from the bot itself are ignored."""
    # Mock bot identity
    channel._bot_user_id = 12345
    
    update = MagicMock()
    update.message.from_user.id = 12345 # Same as bot
    update.effective_user.id = 12345
    
    # Mock bus to ensure no publication happens
    channel.bus.publish_inbound = AsyncMock()
    
    await channel._on_message(update, None)
    
    channel.bus.publish_inbound.assert_not_awaited()
