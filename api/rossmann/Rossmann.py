import pickle 
import inflection
import pandas as pd
import numpy as np
import math
import datetime


class Rossmann( object ):
    def __init__( self ):
        self.competition_distance   = pickle.load( open( 'C:/Users/f0ints/repos/prediction-sell/encoding/competition_distance.pkl', 'rb' ) )
        self.competition_time_month = pickle.load( open( 'C:/Users/f0ints/repos/prediction-sell/encoding/competition_time_month.pkl', 'rb' ) )
        self.promo_time_week        = pickle.load( open( 'C:/Users/f0ints/repos/prediction-sell/encoding/promo_time_week.pkl', 'rb' ) )
        self.store_type             = pickle.load( open( 'C:/Users/f0ints/repos/prediction-sell/encoding/store_type.pkl', 'rb' ) )
        self.year                   = pickle.load( open( 'C:/Users/f0ints/repos/prediction-sell/encoding/year.pkl', 'rb' ) )
        self.store_type                  = pickle.load( open( 'C:/Users/f0ints/repos/prediction-sell/encoding/store_type.pkl', 'rb' ) )

    def data_cleaning( self, df1 ):
   
        cols_old_names = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
       'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
       'Promo2SinceYear', 'PromoInterval']

        #Create a function using the inflection library with undersocore method to puth _ between the name of the columns
        snakecase = lambda x: inflection.underscore(x)

        #Create a list with new columns name
        cols_new_name = list(map(snakecase, cols_old_names))

        #replace the old to new columns names
        df1.columns = cols_new_name


        #Change the date object to datetime type
        df1['date'] = pd.to_datetime( df1['date'] )

        ##### 3.1.4.1 Fillout NA's

        #CompetitionDistance - distance in meters to the nearest competitor store
        #decision: Assuming two hypotheses, or the store doesn't have competition or the competitor store is very far.
        #action: Fillout the NA's with a big distance to assuming these stores doesn't have a competition.
        # df1['competition_distance'].max() = 75860.0

        df1['competition_distance'] = df1['competition_distance'].apply( lambda x: 200000.0 if math.isnan( x ) else x )


        #CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
        #decision: Assuming these hypotheses, or the store doesn't open yet or the store opened before our store or we didn't record when the competitor store was open.
        #action: Fillout the NA's with the same sale date of the same row.

        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan( x['competition_open_since_month'] ) else x['competition_open_since_month'], axis=1)
        df1['competition_open_since_year']  = df1.apply(lambda x: x['date'].year if math.isnan( x['competition_open_since_year'] ) else x['competition_open_since_year'], axis=1)


        #Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
        #Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
        #decision: Assuming the store doesn't want to participate of the promo2 .
        #action: Fillout the NA's with the same sale date of the same row.

        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan( x['promo2_since_week'] ) else x['promo2_since_week'], axis=1)
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan( x['promo2_since_year'] ) else x['promo2_since_year'], axis=1)

        #PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store#decision: Assuming the store doesn't want to participate of the promo2 .
        #decision: Getting the month of the date sale and replace number to prefix month later fillna to 0 and cheching if this month is into the promo_interval, if yes set 1 else 0
        #action: Fillout the NA's with the same sale date of the same row.

        #Create the month_map
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

        #Fillna = 0, implace is not show the message/question
        df1['promo_interval'].fillna(0, inplace=True)

        #Create a column to see the months in "prefix" name
        #get the DATE columns, transform in DATE(dt), get the month (month), and map(map) all data using the variable month_map
        df1['month_map'] = df1['date'].dt.month.map( month_map )

        #First condition: Split the promo_terval in comma, and check if df1['month_map'] is in on df1['promo_interval'], if yes 1 else 0
        # 1 = The store is participating of the promo
        # 0 = The store is not participating of the promo

        #Second condition: If df1['promo_interval'] = 0 means that the store wasn't participating of the consecutive promotion
        df1['ispromo'] = df1[['promo_interval','month_map']].apply (lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split( ',' ) else 0, axis=1)




        #Change types to INT
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype( 'int64' )
        df1['competition_open_since_year']  = df1['competition_open_since_year'].astype( 'int64' )
        df1['promo2_since_week']            = df1['promo2_since_week'].astype( 'int64' )
        df1['promo2_since_year']            = df1['promo2_since_year'].astype( 'int64' )

        return df1

    def feature_engineering( self, df2):


        #day
        df2['day'] = df2['date'].dt.day

        #Year
        df2['year'] = df2['date'].dt.year

        #month
        df2['month'] = df2['date'].dt.month

        #week of year
        df2['week_of_year'] = df2['date'].dt.weekofyear

        #year week
        df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )


        #competition_since
        #Merge the competition_open_since_month and competition_open_since_year
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'] , month=x['competition_open_since_month'] , day=1 ), axis=1 )
        # Divide per 30 days to keep the month type
        df2['competition_time_month'] = ( (df2['date'] - df2['competition_since']) / 30).apply(lambda x: x.days ).astype( 'int64' )


        #promo_since
        #Merge the promo2_since_week and promo2_since_year
        # TRICK #
        #First, concatenate and trandform in string = 2023-31
        df2['promo_since'] = df2['promo2_since_year'].astype( str )+ '-' + df2['promo2_since_week'].astype( str )
        # Deduce 1 day, put in the format 2023-31- and remove 7 days.
        #Strptime python is used to convert string to datetime object
        #https://www.nbshare.io/notebook/510557327/Strftime-and-Strptime-In-Python/
        df2['promo_since'] = df2['promo_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', '%Y-%W-%w' ) - datetime.timedelta( days=7 ) )
        df2['promo_time_week'] = ( ( df2['date'] - df2['promo_since'] ) / 7 ).apply(lambda x: x.days).astype( 'int64' )

        #Assortment
        #a = basic, b = extra, c = extended
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b'else 'extended')

        #state_holiday
        #a = public holiday, b = Easter holiday, c = Christmas, 0 = None
        df2['state_holiday'] = df2['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')


        
        #Exclude if the store is close = 0 and sales need be greather then 0, because if the store is closed the sales will be zero.
        df2 = df2[df2['open'] != 0 ]

        ### 4.4.2 Columns Variables Filter 

        #Need to create other study to predict the total of customers in the next weeks. We can do this on the cicle 2, per example.
        #After filter the column open we can drop.
        #Delete the aux columns on the steps before
        cols_drop = ['open', 'promo_interval', 'month_map']
        df2 = df2.drop( cols_drop, axis = 1)
        
        return df2

    def data_preparation(self, df5 ):

        ## 5.2 Rescaling
        #competition_distance

        df5['competition_distance']   = self.competition_distance.transform( df5[['competition_distance']].values )
        df5['competition_time_month'] = self.competition_time_month.transform( df5[['competition_time_month']].values )
        df5['promo_time_week']        = self.promo_time_week.transform( df5[['promo_time_week']].values )
        df5['year']                   = self.year.transform( df5[['year']].values )

        #state_holiday
        #enconding: One Hot Enconding
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])

        df5['store_type'] = self.store_type.transform(df5['store_type'])


        #assortment
        #enconding: Ordinal Enconding 
        assortment_dict = {'basic':1,
                           'extra':2,
                           'extended':3
                          }
        df5['assortment'] = df5['assortment'].map( assortment_dict )


        #Natural Cyclical Transformation

        #day_of_week.
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2 * np.pi/7 )))
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2 * np.pi/7 )))


        #month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2 * np.pi/12 )))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2 * np.pi/12 )))

        #day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2 * np.pi/30 )))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2 * np.pi/30 )))

        #week_of_year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2 * np.pi/52 )))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2 * np.pi/52 )))

        cols_selected =  ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month',
                             'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month',
                             'promo_time_week', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
                             'week_of_year_sin', 'week_of_year_cos']

        
        return df5[cols_selected]
    
    
    def get_prediction( self, model, original_data, test_data):
        pred = model.predict(test_data)
        
        #join pred into the original data
        
        original_data['prediction'] = np.expm1( pred )
        
        return original_data.to_json(orient='records', date_format='iso')