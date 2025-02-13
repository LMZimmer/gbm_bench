import datetime


def timed_print(print_message):
    time = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[INFO | {time}]: ", print_message)
