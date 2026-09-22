# =====================================================================
#  TRABAJO PRÁCTICO INTEGRADOR R
#  LABORATORIO DE PROGRAMACIÓN EN PYTHON Y R
#  BACKTEST ARIMA - IPC GBA (INDEC)
#  Sebastián Uria Minaberrigaray
# =====================================================================

library(dplyr)
library(tseries)
library(forecast) 

### 1. Dataset ################################################################

url <- "https://www.indec.gob.ar/ftp/cuadros/economia/serie_ipc_aperturas.csv"
ipc <- read.csv2(url, stringsAsFactors = FALSE, fileEncoding = "latin1")

# La base de datos incluye el IPC general y desglosado por apertura comercial
# para cada región (Cuyo, GBA, Noreste, Noroeste, Pampeana, y Patagonia) en  
# cada período. Tenemos que filtrarla para quedarnos solo con el nivel general  
# y elegir una sola región (GBA) para posibilitar el análisis.

sub <- ipc %>%
  dplyr::filter(Descripcion_aperturas == "Nivel general", Region == "GBA") %>%
  dplyr::arrange(Periodo)
sub <- sub[-1, ]

# Convertimos etiqueta del período a formato mes y año
anio0 <- as.integer(substr(sub$Periodo[1], 1, 4))
mes0  <- as.integer(substr(sub$Periodo[1], 5, 6))

# Convertimos df en ts
fipc <- ts(sub$v_m_IPC, start = c(anio0, mes0), frequency = 12)

class(fipc)
print(fipc)

# Marcamos elecciones
dec <- function(anio, mes) anio + (mes - 1) / 12
elec_parl <- c(dec(2017,10), dec(2021,11), dec(2025,10))
elec_ger <- c(dec(2019,10), dec(2023,10), dec(2027,10))

plot(fipc,
     main = "Índice de Precios al Consumidor (IPC)\nvar. mensual ene2016 - ago2026",
     ylab = "Var. mensual (%)", xlab = "Año",
     col = "black", lwd = 1.2)
abline(v = elec_parl, col = "steelblue",  lty = 2, lwd = 1.2)
abline(v = elec_ger,  col = "darkorange", lty = 2, lwd = 1.2)
legend("topleft",
       legend = c("Elecc. parlamentarias", "Elecc. generales"),
       col = c("steelblue", "darkorange"),
       lty = 2, lwd = 1.2, bty = "o", cex = 0.85)


### 2. ARIMA ##################################################################

# ACF PACF ADF
par(mfrow = c(1, 2))
acf(fipc,  main = "ACF - fipc (nivel)")
pacf(fipc, main = "PACF - fipc (nivel)")
adf.test(fipc)                   # Rechazamos H0. Caída muy lenta. Diferenciar!

fipc_diff <- diff(fipc)
acf(fipc_diff,  main = "ACF - Primera diferencia")
pacf(fipc_diff, main = "PACF - Primera diferencia")
adf.test(fipc_diff)              # Mucho mejor!

par(mfrow = c(1, 1))

# Corremos ARIMA
fit_ipc <- auto.arima(fipc)
summary(fit_ipc)

checkresiduals(fit_ipc)


#### 3. Backtesting y forecast ################################################
n_total <- length(fipc)
n_test  <- 42   # holdout hasta antes del pico de 2023-2024
n_test2 <- 6   # holdout después del pico

ipc_train <- head(fipc, n_total - n_test)
ipc_test  <- tail(fipc, n_test)
ipc_train2  <- head(fipc, n_total - n_test2)
ipc_test2   <- tail(fipc, n_test2)

fit_backtest <- auto.arima(ipc_train)
summary(fit_backtest)
checkresiduals(fit_backtest)

#Comparar con random walk
arimarw <- Arima(ipc_train, order = c(0, 1, 0))
summary(arimarw)
checkresiduals(arimarw)

AIC(arimarw, fit_backtest)

forecast_backtest <- forecast(fit_backtest, h = n_test)
stopifnot(all.equal(time(ipc_test), time(forecast_backtest$mean)))

# Graficamos pronóstico del backtesting vs. valores reales
plot(forecast_backtest,
     main = "Backtesting ARIMA: \nproyección IPC vs. valores reales",
     ylab = "Variación mensual IPC", xlab = "Tiempo",
     ylim = c(0, 30))
abline(v = elec_parl, col = "lightgray", lty = 2, lwd = 1.2)
abline(v = elec_ger, col = "darkgrey", lty = 2, lwd = 1.5)
lines(ipc_test, col = "firebrick4", lwd = 2)
legend("topleft",
       legend = c("Proyección", "Valor real (holdout)"),
       col = c("dodgerblue", "firebrick4"),
       lty = 1, lwd = 2, bty = "o", cex = 0.85)

# ¿Qué tanto falla el modelo en predecir el pico + desinflación?
error_arima <- ipc_test - forecast_backtest$mean
cat("Error absoluto medio (MAE) en el holdout:",
    mean(abs(error_arima)), "\n")
cat("Raíz del error cuadrático medio (RMSE) en el holdout:",
    sqrt(mean(error_arima^2)), "\n")

#¿Y después del pico?
fit_backtest2 <- auto.arima(ipc_train2)
summary(fit_backtest2)

checkresiduals(fit_backtest2)     #Básicamente igual al ARIMA completo

forecast_backtest2 <- forecast(fit_backtest2, h = n_test2)
stopifnot(all.equal(time(ipc_test2), time(forecast_backtest2$mean)))

plot(forecast_backtest2,
     main = "Backtesting ARIMA: proyección vs. \nvalores reales (post pico '23-'24)",
     ylab = "Variación mensual IPC", xlab = "Tiempo",
     xlim = c(2024 + 6/12, 2026 + 7/12),
     ylim = c(0, 30))
legend("topleft",
       legend = c("Proyección", "Valor real (holdout)"),
       col = c("dodgerblue", "firebrick4"),
       lty = 1, lwd = 2, bty = "o", cex = 0.85)
lines(ipc_test2, col = "firebrick4", lwd = 2)

error_arima2 <- ipc_test2 - forecast_backtest2$mean
cat("MAE:", mean(abs(error_arima2)), "\n")
cat("RMSE:", sqrt(mean(error_arima2^2)), "\n")

#El pronóstico es más preciso, pero está tan fiteado al pico que los valores
#que predice son prácticamente constantes.



### 4. Forecasting recursivo (mes a mes) ######################################

h          <- 1                    # horizonte: 1 mes adelante
n_forecast <- length(ipc_test)

pronosticos <- rep(NA_real_, n_forecast)
errores     <- rep(NA_real_, n_forecast)

#Loop para forecast mes a mes
for (i in seq_len(n_forecast)) {
  corte <- length(ipc_train) + i - 1
  datos <- head(fipc, corte)          
  
  fit <- auto.arima(datos)
  fc  <- forecast(fit, h = h)
  
  pronosticos[i] <- fc$mean[h]
  errores[i]     <- ipc_test[i] - pronosticos[i]
  
  cat("Mes", i,
      "| Real:",       round(ipc_test[i], 2),
      "| Pronóstico:", round(pronosticos[i], 2),
      "| Error:",      round(errores[i], 2), "\n")
}

pronosticos_ts <- ts(pronosticos,
                     start     = time(ipc_test)[1],
                     frequency = frequency(ipc_test))

stopifnot(all.equal(time(pronosticos_ts), time(ipc_test)))

#Métricas globales
cat("\n--- Resumen ---\n")
cat("MAE: ", round(mean(abs(errores)), 2), "\n")
cat("RMSE:", round(sqrt(mean(errores^2)), 2), "\n")

#Métricas durante y post quiebre 2023
n_quiebre <- 4   # dic-23 a mar-24

cat("\n--- Desempeño por período ---\n")
cat("MAE durante el quiebre:", round(mean(abs(errores[1:n_quiebre])), 2), "\n")
cat("MAE post-quiebre:      ",
    round(mean(abs(errores[(n_quiebre + 1):n_forecast])), 2), "\n")

mae  <- mean(abs(errores))
rmse <- sqrt(mean(errores^2))
mae_pico <- mean(abs(errores[1:n_quiebre]))
mae_post <- mean(abs(errores[(n_quiebre + 1):n_forecast]))

plot(ipc_test, type = "o", col = "darkred", lwd = 2,
     main = "Pronóstico recursivo \nmensual vs. IPC real", 
     ylab = "Var. %", xlab = "Mes")
abline(v = elec_parl, col = "lightgray", lty = 2, lwd = 1.2)
abline(v = elec_ger, col = "darkgrey", lty = 2, lwd = 1.8)
lines(pronosticos_ts, col = "blue", lwd = 2)
legend("topright",
       legend = c("Real",
                  "Pronóstico",
                  sprintf("MAE = %.2f",  mae),
                  sprintf("RMSE = %.2f", rmse),
                  sprintf("MAE quiebre = %.2f", mae_pico),
                  sprintf("MAE post    = %.2f", mae_post)),
       col = c("darkred", "blue", NA, NA),
       lty = c(1, 1, NA, NA, NA, NA),
       lwd = 2, bty = "o", cex = 0.85)



### 5. Y ahora, ¿qué puede pasar? #############################################
forecast_2027 <- forecast(fipc, h = 16)
summary(forecast_2027)
print(forecast_2027)
checkresiduals(forecast_2027)

plot(forecast_2027,
     main = "Variación IPC y proyección hasta 2027",
     ylab = "Variación mensual IPC", xlab = "Tiempo",
     ylim = c(0, 30))
abline(v = elec_parl, col = "lightgray", lty = 2, lwd = 1.2)
abline(v = elec_ger, col = "darkgrey", lty = 2, lwd = 1.5)

legend("topleft",
       legend = c("Proyección"),
       col = c("dodgerblue"),
       lty = 1, lwd = 2, bty = "o", cex = 0.85)

#################################### ~ ########################################
