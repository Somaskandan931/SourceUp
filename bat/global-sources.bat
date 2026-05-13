@echo off
setlocal enabledelayedexpansion

echo ========================================
echo TradeIndia Supplier Scraper v2.0 - AUTO PAGE DETECTION
echo ========================================
echo.

:: ===== INPUT / OUTPUT =====
set INPUT_PATH=D:\PycharmProjects\SourceUp\data\search_query.csv
set OUTPUT_PATH=D:\PycharmProjects\SourceUp\data\outputs\output_full.csv

:: ===== PAGE RANGE - AUTO DETECTION =====
:: Set START_PAGE=1 (always start from page 1)
:: Set END_PAGE=999 (high number - scraper will auto-stop when no more products)
:: The scraper will detect total pages for each query and stop automatically
set START_PAGE=1
set END_PAGE=999

:: ===== SCRAPING OPTIONS =====
set MAX_BUDGET=9999999
set MAX_DELIVERY=999

:: ===== MAX PHONES TO COLLECT =====
:: Set to 0 to disable (faster), or 5-10 if you need phone numbers
set MAX_PHONES=0

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
    echo.
    echo To rebuild JAR: mvn clean package
    echo.
    pause
    exit /b 1
)

:: ===== BUILD CLASSPATH =====
set CLASSPATH=%JAR_PATH%

if exist "D:\PycharmProjects\SourceUp\target\lib" (
    for %%i in ("D:\PycharmProjects\SourceUp\target\lib\*.jar") do (
        set CLASSPATH=!CLASSPATH!;%%i
    )
)

:: ===== CREATE SESSIONS DIRECTORY =====
if not exist "D:\PycharmProjects\SourceUp\sessions" mkdir "D:\PycharmProjects\SourceUp\sessions"

:: ===== CLEAN PREVIOUS DEBUG FOLDER =====
if exist "debug_html" (
    echo Cleaning old debug files...
    rmdir /s /q debug_html 2>nul
)
mkdir debug_html 2>nul

echo.
echo ========================================
echo TradeIndia Scraper Configuration:
echo   - Input:      %INPUT_PATH%
echo   - Output:     %OUTPUT_PATH%
echo   - Pages:      Auto-detected (starting from page %START_PAGE%)
echo   - Max Budget: %MAX_BUDGET% (no limit)
echo   - Max Days:   %MAX_DELIVERY% (no limit)
echo   - Max Phones: %MAX_PHONES% (0 = disabled)
echo   - Queries:    Reading from CSV...
echo ========================================
echo.

:: Count queries
set QUERY_COUNT=0
for /f "skip=1 tokens=1 delims=," %%a in (%INPUT_PATH%) do (
    set /a QUERY_COUNT+=1
)
echo Total queries to scrape: %QUERY_COUNT%
echo.

echo Running TradeIndia scraper...
echo The scraper will automatically detect total pages for each query.
echo It will stop when no more products are found.
echo.

:: ===== RUN =====
java -cp "!CLASSPATH!" com.somas.global_sources_scraper.App "%INPUT_PATH%" "%OUTPUT_PATH%" %MAX_BUDGET% %MAX_DELIVERY% %START_PAGE% %END_PAGE% %MAX_PHONES%

:: ===== ERROR CHECK =====
if errorlevel 1 (
    echo.
    echo [ERROR] Scraper failed with exit code %errorlevel%
    echo.
    echo Debug HTML saved in: debug_html\
    pause
    exit /b 1
)

:: ===== SUCCESS =====
if exist "%OUTPUT_PATH%" (
    echo.
    echo ========================================
    echo [SUCCESS] Scraping completed!
    echo ========================================
    echo Output saved to: %OUTPUT_PATH%
    for %%A in ("%OUTPUT_PATH%") do (
        set FILE_SIZE=%%~zA
        set /a FILE_SIZE_MB=!FILE_SIZE!/1048576
        echo File size: !FILE_SIZE_MB! MB
    )
    
    :: Show row count
    echo.
    echo Counting rows in output...
    for /f %%a in ('type "%OUTPUT_PATH%" ^| find /c /v ""') do set ROW_COUNT=%%a
    set /a DATA_ROWS=ROW_COUNT-1
    echo Total products collected: !DATA_ROWS!
    
) else (
    echo.
    echo [WARNING] No data collected!
    echo Check debug_html/ folder for saved page sources.
)

echo.
echo ========================================
echo Next steps:
echo   1. Run: python pipeline/run_all.py --full
echo   2. Or rebuild features: python features/feature_builder.py
echo ========================================
echo.
pause