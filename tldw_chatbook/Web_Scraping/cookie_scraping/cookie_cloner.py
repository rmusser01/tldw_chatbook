"""
cookie_cloner.py
================

Browser cookie extraction for authenticated web scraping.

This module provides cross-platform functionality to extract cookies
from major web browsers, enabling scrapers to access authenticated
content. Supports Chrome, Firefox, Edge, and Safari with automatic
cookie decryption.

Security Note:
--------------
This module accesses browser cookie stores which may contain
sensitive authentication tokens. Use responsibly and only for
legitimate purposes on sites you have permission to access.

Supported Browsers:
------------------
- Chrome/Chromium (Windows, macOS, Linux)
- Firefox (Windows, macOS, Linux)
- Microsoft Edge (Windows, macOS, Linux)
- Safari (macOS only)

Main Functions:
--------------
- get_cookies(domain, browser): Extract cookies for a domain
- get_chrome_cookies(): Chrome-specific extraction
- get_firefox_cookies(): Firefox-specific extraction
- get_edge_cookies(): Edge-specific extraction
- get_safari_cookies(): Safari-specific extraction

Example:
--------
    # Get cookies for a specific domain from Chrome
    cookies = get_cookies("example.com", browser="chrome")
    
    # Get cookies from all browsers
    all_cookies = get_cookies("example.com", browser="all")
    
    # Use with Playwright
    browser_cookies = [
        {"name": k, "value": v, "domain": ".example.com", "path": "/"}
        for k, v in cookies.items()
    ]
"""
# Imports
from Cryptodome.Cipher import AES
from Cryptodome.Protocol.KDF import PBKDF2
from datetime import datetime
import base64
import datetime
import json
import keyring
import os
import shutil
import sqlite3
import struct
import sys
#
# Third-Party Imports
from loguru import logger
#
########################################################################################################################
#
# Chrome Cookies

def get_chrome_cookies(domain_name):
    """
    Extract cookies from Chrome/Chromium browsers.
    
    Handles cookie decryption across different platforms:
    - Windows: Uses DPAPI and/or AES decryption
    - macOS: Uses Keychain for decryption key
    - Linux: Uses default password 'peanuts'
    
    Args:
        domain_name (str): Domain to filter cookies (e.g., 'example.com')
        
    Returns:
        dict: Cookie names mapped to values
        
    Note:
        Requires browser to be closed for database access
    """
    global win32crypt
    if sys.platform == 'win32':
        import win32crypt
        from Cryptodome.Cipher import AES
        appdata_path = os.getenv('LOCALAPPDATA')
        cookie_path = os.path.join(appdata_path, r'Google\Chrome\User Data\Default\Cookies')
        local_state_path = os.path.join(appdata_path, r'Google\Chrome\User Data\Local State')
    elif sys.platform == 'darwin':
        from Cryptodome.Cipher import AES
        import keyring
        cookie_path = os.path.expanduser('~/Library/Application Support/Google/Chrome/Default/Cookies')
        local_state_path = os.path.expanduser('~/Library/Application Support/Google/Chrome/Local State')
    else:  # Linux
        cookie_path = os.path.expanduser('~/.config/google-chrome/Default/Cookies')
        local_state_path = os.path.expanduser('~/.config/google-chrome/Local State')

    # Read the encryption key
    with open(local_state_path, 'r', encoding='utf-8') as f:
        local_state = json.load(f)
    encrypted_key = base64.b64decode(local_state['os_crypt']['encrypted_key'])
    encrypted_key = encrypted_key[5:]  # Remove 'DPAPI' prefix

    if sys.platform == 'win32':
        import win32crypt
        key = win32crypt.CryptUnprotectData(encrypted_key, None, None, None, 0)[1]
    elif sys.platform == 'darwin':
        # Decrypt using Keychain
        from Cryptodome.Protocol.KDF import PBKDF2
        import keyring
        import hashlib

        password = keyring.get_password("Chrome Safe Storage", "Chrome")
        if password is None:
            password = "peanuts"
        iterations = 1003
        salt = b'saltysalt'
        key = PBKDF2(password.encode('utf-8'), salt, dkLen=16, count=iterations)
    else:  # Linux
        # On Linux, Chrome uses the 'peanuts' password by default
        from Cryptodome.Protocol.KDF import PBKDF2
        import hashlib

        password = 'peanuts'
        iterations = 1
        salt = b'saltysalt'
        key = PBKDF2(password.encode('utf-8'), salt, dkLen=16, count=iterations)

    # Copy the Cookies file to avoid locking the database
    temp_cookie_path = os.path.join(os.getcwd(), 'chrome_cookies_temp')
    shutil.copyfile(cookie_path, temp_cookie_path)

    conn = sqlite3.connect(temp_cookie_path)
    cursor = conn.cursor()

    cookies = {}
    try:
        cursor.execute("SELECT host_key, name, path, encrypted_value, expires_utc FROM cookies WHERE host_key LIKE ?",
                       ('%' + domain_name + '%',))
        for host_key, name, path, encrypted_value, expires_utc in cursor.fetchall():
            if sys.platform == 'win32':
                try:
                    decrypted_value = win32crypt.CryptUnprotectData(encrypted_value, None, None, None, 0)[1]
                except:
                    decrypted_value = decrypt_edge_cookie(encrypted_value, key)
            else:
                decrypted_value = decrypt_edge_cookie(encrypted_value, key)

            expires = datetime.datetime(1601, 1, 1) + datetime.timedelta(microseconds=expires_utc)
            if expires < datetime.datetime.now():
                continue  # Skip expired cookies

            cookies[name] = decrypted_value.decode('utf-8', 'ignore')
    finally:
        cursor.close()
        conn.close()
        os.remove(temp_cookie_path)

    return cookies

def decrypt_chrome_cookie(encrypted_value, key):
    # Remove prefix (v10 or v11)
    if encrypted_value[:3] == b'v10' or encrypted_value[:3] == b'v11':
        encrypted_value = encrypted_value[3:]
    iv = b' ' * 16
    cipher = AES.new(key, AES.MODE_CBC, IV=iv)
    decrypted = cipher.decrypt(encrypted_value)
    # Remove PKCS#7 padding
    padding_length = decrypted[-1]
    return decrypted[:-padding_length]


#
# End of Chrome Cookies
########################################################################################################################


########################################################################################################################
#
# Firefox Cookies

def get_firefox_cookies(domain_name):
    """
    Extract cookies from Firefox browser.
    
    Firefox stores cookies in an SQLite database without encryption,
    making extraction simpler than Chromium-based browsers.
    
    Args:
        domain_name (str): Domain to filter cookies
        
    Returns:
        dict: Cookie names mapped to values
        
    Note:
        Searches for default-release profile
    """
    if sys.platform == 'win32':
        appdata_path = os.getenv('APPDATA')
        profile_path = os.path.join(appdata_path, 'Mozilla', 'Firefox', 'Profiles')
    elif sys.platform == 'darwin':
        profile_path = os.path.expanduser('~/Library/Application Support/Firefox/Profiles')
    else:  # Linux
        profile_path = os.path.expanduser('~/.mozilla/firefox')

    profiles = [os.path.join(profile_path, p) for p in os.listdir(profile_path) if p.endswith('.default-release')]
    cookies = {}

    for profile in profiles:
        cookie_db = os.path.join(profile, 'cookies.sqlite')
        if not os.path.exists(cookie_db):
            continue

        temp_cookie_db = os.path.join(os.getcwd(), 'firefox_cookies_temp')
        shutil.copyfile(cookie_db, temp_cookie_db)

        conn = sqlite3.connect(temp_cookie_db)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT host, name, value, expiry FROM moz_cookies WHERE host LIKE ?",
                           ('%' + domain_name + '%',))
            for host, name, value, expiry in cursor.fetchall():
                expires = datetime.datetime.fromtimestamp(expiry)
                if expires < datetime.datetime.now():
                    continue  # Skip expired cookies
                cookies[name] = value
        finally:
            cursor.close()
            conn.close()
            os.remove(temp_cookie_db)
            break  # Use the first profile found

    return cookies


#
# End of Firefox Cookies
########################################################################################################################


########################################################################################################################
#
# MS Edge Cookies

def get_edge_cookies(domain_name):
    """
    Extract cookies from Microsoft Edge browser.
    
    Uses similar decryption methods as Chrome since Edge is
    Chromium-based, but with different storage locations.
    
    Args:
        domain_name (str): Domain to filter cookies
        
    Returns:
        dict: Cookie names mapped to values
        
    Raises:
        FileNotFoundError: If Edge cookie files not found
    """
    try:
        if sys.platform == 'win32':
            import win32crypt
            appdata_path = os.getenv('LOCALAPPDATA')
            cookie_path = os.path.join(appdata_path, r'Microsoft\Edge\User Data\Default\Cookies')
            local_state_path = os.path.join(appdata_path, r'Microsoft\Edge\User Data\Local State')
        elif sys.platform == 'darwin':
            import keyring
            cookie_path = os.path.expanduser('~/Library/Application Support/Microsoft Edge/Default/Cookies')
            local_state_path = os.path.expanduser('~/Library/Application Support/Microsoft Edge/Local State')
        else:  # Linux
            cookie_path = os.path.expanduser('~/.config/microsoft-edge/Default/Cookies')
            local_state_path = os.path.expanduser('~/.config/microsoft-edge/Local State')

        # Ensure the cookie and local state files exist
        if not os.path.exists(cookie_path):
            raise FileNotFoundError(f"Cookie file not found at {cookie_path}")
        if not os.path.exists(local_state_path):
            raise FileNotFoundError(f"Local State file not found at {local_state_path}")

        # Read and decode the encryption key
        with open(local_state_path, 'r', encoding='utf-8') as f:
            local_state = json.load(f)

        encrypted_key = base64.b64decode(local_state['os_crypt']['encrypted_key'])

        if sys.platform == 'win32':
            # Remove 'DPAPI' prefix
            encrypted_key = encrypted_key[5:]
            key = win32crypt.CryptUnprotectData(encrypted_key, None, None, None, 0)[1]
        elif sys.platform == 'darwin':
            # macOS Keychain access
            password = keyring.get_password("Microsoft Edge Safe Storage", "Microsoft Edge")
            if password is None:
                password = "peanuts"  # Default password for Chromium-based browsers
            iterations = 1003
            salt = b'saltysalt'
            key = PBKDF2(password.encode('utf-8'), salt, dkLen=16, count=iterations)
        else:  # Linux
            # Linux may use 'peanuts' or libsecret for keyring
            password = 'peanuts'
            iterations = 1
            salt = b'saltysalt'
            key = PBKDF2(password.encode('utf-8'), salt, dkLen=16, count=iterations)

        # Copy the Cookies file to avoid database lock
        temp_cookie_path = os.path.join(os.getcwd(), 'edge_cookies_temp')
        shutil.copyfile(cookie_path, temp_cookie_path)

        conn = sqlite3.connect(temp_cookie_path)
        cursor = conn.cursor()

        cookies = {}
        try:
            cursor.execute("""
                SELECT host_key, name, path, encrypted_value, expires_utc
                FROM cookies WHERE host_key LIKE ?
            """, ('%' + domain_name + '%',))

            for host_key, name, path, encrypted_value, expires_utc in cursor.fetchall():
                if sys.platform == 'win32':
                    try:
                        # Try to decrypt using CryptUnprotectData
                        decrypted_value = win32crypt.CryptUnprotectData(encrypted_value, None, None, None, 0)[1]
                    except:
                        # If failed, use custom decryption
                        decrypted_value = decrypt_edge_cookie(encrypted_value, key)
                else:
                    decrypted_value = decrypt_edge_cookie(encrypted_value, key)

                # Convert timestamp to datetime
                expires = datetime.datetime(1601, 1, 1) + datetime.timedelta(microseconds=expires_utc)
                if expires < datetime.datetime.now():
                    continue  # Skip expired cookies

                cookies[name] = decrypted_value.decode('utf-8', 'ignore')
        finally:
            cursor.close()
            conn.close()
            os.remove(temp_cookie_path)

        return cookies

    except Exception as e:
        logger.error(f"An error occurred while retrieving Edge cookies: {e}")
        return {}

def decrypt_edge_cookie(encrypted_value, key):
    # Remove 'v10' or 'v11' prefix
    if encrypted_value[:3] in (b'v10', b'v11'):
        encrypted_value = encrypted_value[3:]

    # Set up the cipher
    iv = b' ' * 16  # 16 spaces
    cipher = AES.new(key, AES.MODE_CBC, IV=iv)

    decrypted = cipher.decrypt(encrypted_value)

    # Remove padding
    padding_length = decrypted[-1]
    return decrypted[:-padding_length]


#
# End of MS Edge Cookies
########################################################################################################################


########################################################################################################################
#
# Safari Cookies

def get_safari_cookies(domain_name):
    """
    Extract cookies from Safari browser (macOS only).
    
    Parses Safari's binary cookie format (.binarycookies).
    
    Args:
        domain_name (str): Domain to filter cookies
        
    Returns:
        dict: Cookie names mapped to values
        
    Note:
        Only available on macOS
    """
    cookie_file_paths = [
        os.path.expanduser('~/Library/Cookies/Cookies.binarycookies'),
        os.path.expanduser('~/Library/Containers/com.apple.Safari/Data/Library/Cookies/Cookies.binarycookies')
    ]

    cookies = {}

    for cookie_file in cookie_file_paths:
        if os.path.exists(cookie_file):
            with open(cookie_file, 'rb') as f:
                magic = f.read(4)
                if magic != b'cook':
                    raise Exception('Not a Cookies.binarycookies file')
                num_pages = struct.unpack('>i', f.read(4))[0]
                page_sizes = []
                for _ in range(num_pages):
                    page_sizes.append(struct.unpack('>i', f.read(4))[0])

                for page_size in page_sizes:
                    page_data = f.read(page_size)
                    cookies.update(parse_safari_page(page_data, domain_name))

    return cookies

def parse_safari_page(page_data, domain_name):
    cookies = {}
    page_header = page_data[:4]
    num_cookies = struct.unpack('>i', page_data[4:8])[0]
    cookie_offsets = []
    for i in range(num_cookies):
        offset = struct.unpack('>i', page_data[8 + i*4:12 + i*4])[0]
        cookie_offsets.append(offset)
    for offset in cookie_offsets:
        cookie = parse_safari_cookie(page_data[offset:], domain_name)
        if cookie:
            cookies[cookie['name']] = cookie['value']
    return cookies

def parse_safari_cookie(data, domain_name):
    try:
        flags = struct.unpack('<i', data[0:4])[0]
        url_offset = struct.unpack('<i', data[4:8])[0]
        name_offset = struct.unpack('<i', data[8:12])[0]
        path_offset = struct.unpack('<i', data[12:16])[0]
        value_offset = struct.unpack('<i', data[16:20])[0]
        end_of_cookie = data.find(b'\x00', value_offset)
        cookie_name = data[name_offset:data.find(b'\x00', name_offset)].decode('utf-8')
        cookie_value = data[value_offset:end_of_cookie].decode('utf-8')
        cookie_domain = data[url_offset:data.find(b'\x00', url_offset)].decode('utf-8')
        if domain_name in cookie_domain:
            return {'name': cookie_name, 'value': cookie_value}
    except Exception as e:
        pass
    return None

#
# End of Safari Cookies
########################################################################################################################


########################################################################################################################
#
# Main Function

def get_cookies(domain_name, browser='all'):
    """
    Extract cookies for a domain from specified browser(s).
    
    Main entry point for cookie extraction. Attempts to extract
    cookies from specified browser(s) and combines results.
    
    Args:
        domain_name (str): Domain to extract cookies for
        browser (str): Browser to extract from:
            - 'all': Try all available browsers
            - 'chrome': Chrome/Chromium only
            - 'firefox': Firefox only
            - 'edge': Microsoft Edge only
            - 'safari': Safari only (macOS)
            
    Returns:
        dict: Combined cookies from all specified browsers
            Format: {cookie_name: cookie_value, ...}
            
    Example:
        >>> cookies = get_cookies("github.com", "chrome")
        >>> print(f"Found {len(cookies)} cookies")
        
    Note:
        - Errors are logged but don't stop execution
        - Returns empty dict if no cookies found
        - Browsers must be closed for reliable access
    """
    cookies = {}

    if browser in ('all', 'chrome'):
        # Get cookies from Chrome
        try:
            chrome_cookies = get_chrome_cookies(domain_name)
            cookies.update(chrome_cookies)
        except Exception as e:
            logger.error(f"Failed to get Chrome cookies: {e}")

    if browser in ('all', 'firefox'):
        # Get cookies from Firefox
        try:
            firefox_cookies = get_firefox_cookies(domain_name)
            cookies.update(firefox_cookies)
        except Exception as e:
            logger.error(f"Failed to get Firefox cookies: {e}")

    if browser in ('all', 'edge'):
        # Get cookies from Edge
        try:
            edge_cookies = get_edge_cookies(domain_name)
            cookies.update(edge_cookies)
        except Exception as e:
            logger.error(f"Failed to get Edge cookies: {e}")

    if sys.platform == 'darwin' and browser in ('all', 'safari'):
        # Get cookies from Safari (macOS only)
        try:
            safari_cookies = get_safari_cookies(domain_name)
            cookies.update(safari_cookies)
        except Exception as e:
            logger.error(f"Failed to get Safari cookies: {e}")

    if not cookies:
        logger.error(f"No cookies found for {domain_name} using browser: {browser}")

    return cookies

#
# End of Main Function
########################################################################################################################
