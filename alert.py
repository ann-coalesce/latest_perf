import credentials
import requests

def send_notif(message, chat_id='-4972938924'):
    bot_key = credentials.DATA_VALIDATION_TELEGRAM_BOT
    
    send_message_url = f'https://api.telegram.org/bot{bot_key}/sendMessage?chat_id={chat_id}&text={message}'
    res = requests.post(send_message_url)
    # print(res.text)
    if res.ok:
        print('Message sent successfully!')
    else:
        print('message not sent')