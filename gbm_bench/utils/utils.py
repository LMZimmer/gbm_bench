import datetime


def timed_print(print_message):
    time = str(datetime.datetime.now().time())
    print(f"{time}: ", print_message)
