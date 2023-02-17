import datetime

# Convert string to datetime object
def convert_to_datetime(date):
    try:
        date_object = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f%z')
    except:
        date_object = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z')
    return date_object

def get_time_difference_as_number(date1, date2):
    date1 = convert_to_datetime(date1)
    date2 = convert_to_datetime(date2)
    return (date2 - date1).total_seconds()