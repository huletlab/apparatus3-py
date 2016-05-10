import  pywapi
import string


def get_dewpoint_f():
        noaa_result = pywapi.get_weather_from_noaa('KHOU')
        return  float(noaa_result['dewpoint_f'])

if __name__ == "__main__":
        print "Dew point in KHOU is (F):",get_dewpoint_f()
