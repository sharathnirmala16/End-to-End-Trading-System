commands = {
    "CreateExchangeTable": """
        CREATE TABLE IF NOT EXISTS Exchange (
            name VARCHAR(255) PRIMARY KEY,
            abbreviation VARCHAR(32) NOT NULL,  
            website_url VARCHAR(255) NULL, 
            city VARCHAR(255) NULL, 
            country VARCHAR(255) NULL, 
            currency VARCHAR(64) NULL, 
            time_zone_offset TIME NULL, 
            created_datetime TIMESTAMP NOT NULL, 
            last_updated_datetime TIMESTAMP NOT NULL
        );
    """,
    "CreateDataVendorTable": """
        CREATE TABLE IF NOT EXISTS DataVendor (
            name VARCHAR(255) PRIMARY KEY,
            website_url VARCHAR(255) NULL,  
            support_email VARCHAR(255) NULL,  
            created_datetime TIMESTAMP NOT NULL, 
            last_updated_datetime TIMESTAMP NOT NULL
        );
    """,
    "CreateSymbolTable": """
        CREATE TABLE IF NOT EXISTS Symbol ( 
            ticker VARCHAR(64) NOT NULL,
            vendor_ticker VARCHAR(64) NOT NULL,
            exchange VARCHAR(255) NOT NULL,
            vendor VARCHAR(255) NOT NULL,
            instrument VARCHAR(64) NOT NULL,
            name VARCHAR(255) NULL,
            sector VARCHAR(255) NULL,  
            interval BIGINT NOT NULL,
            linked_table_name VARCHAR(255) NOT NULL, 
            created_datetime TIMESTAMP NOT NULL, 
            last_updated_datetime TIMESTAMP NOT NULL, 
            PRIMARY KEY (ticker, vendor, exchange, interval),
            CONSTRAINT exchange_frk
                FOREIGN KEY(exchange)
                    REFERENCES Exchange(name),
            CONSTRAINT vendor_frk
                FOREIGN KEY(vendor)
                    REFERENCES DataVendor(name)
        );
    """,
}
