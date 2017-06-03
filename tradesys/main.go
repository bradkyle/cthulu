package main

import (
	"github.com/thrasher-/gocryptotrader/exchanges/poloniex"
	"github.com/thrasher-/gocryptotrader/config"
	"github.com/thrasher-/gocryptotrader/portfolio"
	"github.com/thrasher-/gocryptotrader/exchanges"
	"github.com/thrasher-/gocryptotrader/exchanges/ticker"
	"github.com/thrasher-/gocryptotrader/common"
	"log"
	"os"
	"os/signal"
	"syscall"
	"runtime"
	"strconv"
	"github.com/thrasher-/gocryptotrader/currency"
)

//=====================================================================================================================>
// Exchange/Bot related functions
//=====================================================================================================================>

var bot Bot

//Initialize all of the exchanges that will be used
type ExchangeMain struct {
	poloniex      poloniex.Poloniex
}

type Bot struct {
	config    *config.Config
	portfolio *portfolio.PortfolioBase
	exchange  ExchangeMain
	exchanges []exchange.IBotExchange
	tickers   []ticker.Ticker
	shutdown  chan bool
}

//
func SeedExchangeAccountInfo(data []exchange.ExchangeAccountInfo) {
	if len(data) == 0 {
		return
	}

	port := portfolio.GetPortfolio()

	for i := 0; i < len(data); i++ {
		//get the exchange name
		exchangeName := data[i].ExchangeName

		//
		for j := 0; j < len(data[i].Currencies); j++ {
			//get
			currencyName := data[i].Currencies[j].CurrencyName

			onHold := data[i].Currencies[j].Hold
			avail := data[i].Currencies[j].TotalValue
			total := onHold + avail

			if total <= 0 {
				continue
			}

			if !port.ExchangeAddressExists(exchangeName, currencyName) {
				port.Addresses = append(port.Addresses, portfolio.PortfolioAddress{Address: exchangeName, CoinType: currencyName, Balance: total, Decscription: portfolio.PORTFOLIO_ADDRESS_EXCHANGE})
			} else {
				port.UpdateExchangeAddressBalance(exchangeName, currencyName, total)
			}
		}
	}
}

type AllEnabledExchangeAccounts struct {
	Data []exchange.ExchangeAccountInfo `json:"data"`
}

//Retrieves all enabled/valid exchange accounts
func GetAllEnabledExchangeAccountInfo() AllEnabledExchangeAccounts {

	var response AllEnabledExchangeAccounts

	for _, individualBot := range bot.exchanges {

		if individualBot != nil && individualBot.IsEnabled() {

			individualExchange, err := individualBot.GetExchangeAccountInfo()
			if err != nil {
				log.Println("Error encountered retrieving exchange account for '" + individualExchange.ExchangeName + "'")
			}
			response.Data = append(response.Data, individualExchange)
		}
	}
	return response
}

//=====================================================================================================================>
//Main
//=====================================================================================================================>

func main() {

	HandleInterrupt()

	 //load config using default config file -> config.dat
	bot.config = &config.Cfg
	log.Printf("Loading config file %s..\n", config.CONFIG_FILE)
	err := bot.config.LoadConfig("config_example.dat")
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Bot '%s' started.\n", bot.config.Name)

	//adjust the maximum goroutines that are allowed
	AdjustGoMaxProcs()

	log.Printf("Available Exchanges: %d. Enabled Exchanges: %d.\n", len(bot.config.Exchanges), bot.config.GetConfigEnabledExchanges())
	log.Println("Bot Exchange support:")

	//initialize all exchanges for use in the program
	bot.exchanges = []exchange.IBotExchange{
		new(poloniex.Poloniex),
	}

	//loop though the initialized bot exchanges above setting the default config of each
	for i := 0; i < len(bot.exchanges); i++ {
		if bot.exchanges[i] != nil {
			//sets default attributes used by exchanges
			bot.exchanges[i].SetDefaults()
			log.Printf("Exchange %s successfully set default settings.\n", bot.exchanges[i].GetName())
		}
	}

	//for each exchange in IBotExchange interface
	for _, exch := range bot.config.Exchanges {

		//for each exchange in IBotExchange interface
		for i := 0; i < len(bot.exchanges); i++ {

			//check that the initialization is not empty
			if bot.exchanges[i] != nil {


				if bot.exchanges[i].GetName() == exch.Name {

					//changes default values initialized by SetDefaults to real values
					bot.exchanges[i].Setup(exch)

					if bot.exchanges[i].IsEnabled() {
						log.Printf("%s: Exchange support: %s (Authenticated API support: %s - Verbose mode: %s).\n", exch.Name, common.IsEnabled(exch.Enabled), common.IsEnabled(exch.AuthenticatedAPISupport), common.IsEnabled(exch.Verbose))
						//start a
						bot.exchanges[i].Start()
					} else {
						log.Printf("%s: Exchange support: %s\n", exch.Name, common.IsEnabled(exch.Enabled))
					}


				}
			}

		}
	}

	//
	bot.config.RetrieveConfigCurrencyPairs()

	//seeds currency data relating to base currencies such as USD,HKD,EUR,CAD,AUD,SGD,JPY,GBP,NZD
	err = currency.SeedCurrencyData(currency.BaseCurrencies)
	if err != nil {
		//exits with a fatal error if unable to retrieve currency data
		log.Fatalf("Fatal error retrieving config currencies. Error: %s", err)
	}
	log.Println("Successfully retrieved config currencies.")


	bot.portfolio = &portfolio.Portfolio
	bot.portfolio.SeedPortfolio(bot.config.Portfolio)
	SeedExchangeAccountInfo(GetAllEnabledExchangeAccountInfo().Data)

	go portfolio.StartPortfolioWatcher()

	if !bot.config.Webserver.Enabled {
		log.Println("HTTP Webserver support disabled.")
	}

	<-bot.shutdown
	Shutdown()
}

//=====================================================================================================================>
// Utility functions
//=====================================================================================================================>

func AdjustGoMaxProcs() {
	log.Println("Adjusting bot runtime performance..")
	maxProcsEnv := os.Getenv("GOMAXPROCS")
	maxProcs := runtime.NumCPU()
	log.Println("Number of CPU's detected:", maxProcs)

	if maxProcsEnv != "" {
		log.Println("GOMAXPROCS env =", maxProcsEnv)
		env, err := strconv.Atoi(maxProcsEnv)

		if err != nil {
			log.Println("Unable to convert GOMAXPROCS to int, using", maxProcs)
		} else {
			maxProcs = env
		}
	}
	log.Println("Set GOMAXPROCS to:", maxProcs)
	runtime.GOMAXPROCS(maxProcs)
}


func HandleInterrupt() {
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		sig := <-c
		log.Printf("Captured %v.", sig)
		Shutdown()
	}()
}

func Shutdown() {
	log.Println("Bot shutting down..")
	bot.config.Portfolio = portfolio.Portfolio
	err := bot.config.SaveConfig("")

	if err != nil {
		log.Println("Unable to save config.")
	} else {
		log.Println("Config file saved successfully.")
	}

	log.Println("Exiting.")
	os.Exit(1)
}

