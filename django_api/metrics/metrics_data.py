from prophet.plot import performance_metrics, plot_cross_validation_metric
import matplotlib.pyplot as plt
import pandas as pd

class Metrics:
    def __init__(self, df_cv):
        self.metrics: pd.DataFrame = df_cv  
        self.df_performance: pd.DataFrame = performance_metrics(df_cv)
        
    def save_metrcis(self):
        print(self.df_performance)
        df = self.df_performance 
        df.to_json('dataframe.json', orient='records', date_format='iso')
        
    def show_comparison(self, model):
    
        df_cv = self.metrics
        
        df_cv['yhat'] = model.predict(df_cv)['yhat']
        df_cv['y'] = df_cv['y'].fillna(0)

        plt.figure(figsize=(12, 6))
        plt.plot(df_cv['ds'], df_cv['y'], 'o-', label='Valores Reales', color='yellow')
        plt.plot(df_cv['ds'], df_cv['yhat'], 'x-', label='Valores Predichos', color='blue')

        plt.title('Valores Reales vs Predichos durante Cross Validation')
        plt.xlabel('Fecha')
        plt.ylabel('Valores')
        plt.legend()
        
        plt.savefig('comparison.png')
        plt.show()
        
    def show_MAPE(self):
        
        fig = plot_cross_validation_metric(self.metrics, metric='mape')

        for line in fig.gca().get_lines():
            line.set_markerfacecolor('black')
            line.set_markeredgecolor('black')
            line.set_alpha(0.8)

        plt.title('MAPE - Mean Absolute Percentage Error')
        plt.xlabel('Horizon (Hours)')
        plt.ylabel('MAPE')
        plt.legend(['MAPE'])

        plt.savefig('mape.png')
        plt.show()
        
    def show_MSE(self):
        
        fig, ax = plt.subplots()
        self.df_performance.plot(x='horizon', y='mse', ax=ax, marker='o', linestyle='-', color='blue')
        ax.set_title('MSE - Mean Squared Error')
        ax.set_xlabel('Horizon (Hours)')
        ax.set_ylabel('MSE')
        ax.legend(['MSE'])
        plt.show()
        
    def show_RMSE(self):
        
        fig, ax = plt.subplots()
        self.df_performance.plot(x='horizon', y='rmse', ax=ax, marker='o', linestyle='-', color='blue')
        ax.set_title('RMSE - Root Mean Squared Error')
        ax.set_xlabel('Horizon (Hours)')
        ax.set_ylabel('RMSE')
        ax.legend(['RMSE'])
        
        plt.savefig('rmse.png')
        plt.show()
        
    def show_MDAPE(self):
        
        fig, ax = plt.subplots()
        self.df_performance.plot(x='horizon', y='mdape', ax=ax, marker='o', linestyle='-', color='blue')
        ax.set_title('MDAPE - Median Absolute Percentage Error')
        ax.set_xlabel('Horizon (Hours)')
        ax.set_ylabel('MDAPE')
        ax.legend(['MDAPE'])
        
        plt.savefig('mdape.png')
        plt.show()
        
    def show_SMAPE(self):
        
        fig, ax = plt.subplots()
        self.df_performance.plot(x='horizon', y='smape', ax=ax, marker='o', linestyle='-', color='blue')
        ax.set_title('SMAPE - Symetric Absolute Percentage Error')
        ax.set_xlabel('Horizon (Hours)')
        ax.set_ylabel('SMAPE')
        ax.legend(['SMAPE'])
        
        plt.savefig('smape.png')
        plt.show()
       
    def show_Coverage(self):
        
        fig, ax = plt.subplots()
        self.df_performance.plot(x='horizon', y='coverage', ax=ax, marker='o', linestyle='-', color='blue')
        ax.set_title('Coverage')
        ax.set_xlabel('Horizon (Hours)')
        ax.set_ylabel('Coverage')
        ax.legend(['Coverage'])
        
        plt.savefig('coverage.png')
        plt.show()