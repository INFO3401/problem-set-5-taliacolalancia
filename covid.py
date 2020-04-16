# Utility functions for epidemiology data
import pandas as pd

# Use this to change any date information to the right type
# and to move all date data from columns to rows
def correctDateFormat(df):
    # Move date data from columns to rows. Will create two new columns (one for
    # date and one for the number of confirmed cases). Will add a new row for
    # each date x province/state
    df = df.melt(id_vars=df.columns[0:4], var_name="Date", value_name="Confirmed")

    # Convert date to a datetime object so pandas knows how to do math with it
    df["Date"] = pd.to_datetime(df["Date"])
    return df

# Helper function you can use to group the data for a given country of interestself.
# Just pass the function your dataframe and the country's name as a string
def aggregateCountry(df, country):
    data = df.loc[df["Country/Region"] == country]
    return data.groupby("Date", as_index=False).sum()

#15 Problem Set 5
def topCorrelation(dataset, target, number):
    dataset2 = dataset.pivot_table(values=target, index='Date', columns='Country/Region', aggfunc='first')
    dataset2.corr()
    countries = dataset2.columns
    repeatList = []
    repeatList2 = []

    for country1 in countries:
        for country2 in countries:
            if country1 != country2:
                if[country1,country2] not in repeatList and [country2, country1] not in repeatList:
                    repeatList.append([country1, country2])
                    repeatList2.append(dataset2[country1][country2])

        dataset3 = pd.DataFrame(list(zip(repeatList, repeatList2)), coulmns=['Pairs', 'Corr'])
        dataset3.sort_values(by='Corr', inplace=True, ascending=False)
        return(dataset3.iloc[:number])
