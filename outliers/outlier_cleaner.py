#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    errors = predictions-net_worths
    cleaned_data = sorted(zip(ages, net_worths, errors), key=lambda x:abs(x[2]))[0:int(len(ages)*0.9)]

    return cleaned_data

