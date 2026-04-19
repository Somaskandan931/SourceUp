@echo off
setlocal enabledelayedexpansion

echo ========================================
echo TradeIndia Supplier Scraper v1.4
echo ========================================
echo.

:: ===== INPUT / OUTPUT =====
set INPUT_PATH=D:\PycharmProjects\SourceUp\data\search_query.csv
set OUTPUT_PATH=D:\PycharmProjects\SourceUp\data\outputs\output14.csv

:: ===== PAGE RANGE =====
:: TradeIndia only has pages 1-10. Edit START_PAGE and END_PAGE below.
::
::   START_PAGE=1  END_PAGE=5   scrapes pages 1,2,3,4,5  (~140 cards per query)
::   START_PAGE=1  END_PAGE=10  scrapes pages 1..10      (~280 cards per query)
::   START_PAGE=3  END_PAGE=7   scrapes pages 3,4,5,6,7  (~140 cards per query)
::
:: NOTE: The scraper stops exactly at END_PAGE and never goes beyond page 10.
set START_PAGE=1
set END_PAGE=5

:: ===== CHECK INPUT =====
if not exist "%INPUT_PATH%" (
    echo [ERROR] Input file not found: %INPUT_PATH%
    pause
    exit /b 1
)

:: ===== ENSURE OUTPUT FOLDER EXISTS =====
for %%F in ("%OUTPUT_PATH%") do (
    if not exist "%%~dpF" mkdir "%%~dpF"
)

echo Input File  : %INPUT_PATH%
echo Output File : %OUTPUT_PATH%
echo.

:: ===== JAR PATH =====
set JAR_PATH=D:\PycharmProjects\SourceUp\somasjar.jar

if not exist "%JAR_PATH%" (
    echo [ERROR] JAR file not found: %JAR_PATH%
    echo Please rebuild your JAR with the TradeIndia code.
    pause
    exit /b 1
)

:: ===== BUILD CLASSPATH =====
set CLASSPATH=%JAR_PATH%

if exist "D:\PycharmProjects\SourceUp\lib" (
    for %%i in ("D:\PycharmProjects\SourceUp\lib\*.jar") do (
        set CLASSPATH=!CLASSPATH!;%%i
    )
)

:: ===== CREATE SESSIONS DIRECTORY =====
if not exist "D:\PycharmProjects\SourceUp\sessions" mkdir "D:\PycharmProjects\SourceUp\sessions"

echo.
echo ========================================
echo TradeIndia Scraper Configuration:
echo   - Base URL: https://www.tradeindia.com/search.html
echo   - Queries  : No expansion - base queries only
echo   - Page range: %START_PAGE% to %END_PAGE%
echo   - Max pages per query: TradeIndia cap = 10
echo ========================================
echo.

echo Running TradeIndia scraper...
echo.

:: ===== RUN =====
java -cp "!CLASSPATH!" com.somas.global_sources_scraper.App "%INPUT_PATH%" "%OUTPUT_PATH%" 9999999 999 %START_PAGE% %END_PAGE%

:: ===== ERROR CHECK =====
if errorlevel 1 (
    echo.
    echo [ERROR] Scraper failed!
    echo.
    echo Possible issues:
    echo   1. JAR file needs to be rebuilt with TradeIndia code
    echo   2. Network connection problem
    echo   3. ChromeDriver version mismatch
    echo.
    pause
    exit /b 1
)

:: ===== SUCCESS =====
if exist "%OUTPUT_PATH%" (
    echo.
    echo [SUCCESS] Output saved to: %OUTPUT_PATH%
    for %%A in ("%OUTPUT_PATH%") do echo File size: %%~zA bytes
) else (
    echo.
    echo [WARNING] No data collected
    echo Check debug_html/ folder for saved page sources.
)

echo.
pause