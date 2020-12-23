import datetime
import numpy as np

start_date = '2005-01-03'
end_date = '2020-11-30'
date_format = '%Y-%m-%d'
start_datetime = datetime.datetime.strptime(start_date, date_format)
end_datetime = datetime.datetime.strptime(end_date, date_format)
number_datetime = (end_datetime - start_datetime).days + 1

def index_to_date(index,timestamp):
    """
    Args:
        index: the number index
        timestamp: a list of times [2005-01-03, 2005-01-04, ...]
    Returns:
    """
    return timestamp[index]


def date_to_index(date_string,timestamp):
    """
    Args:
        date_string: in format of '2005-01-03'
    Returns: the trade days from start_date: '2005-01-03'
    """
    assert date_string in timestamp, '%s is not a trading day' %(date_string)
    return timestamp.index(date_string)
