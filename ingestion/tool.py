"""
Web Tools Module
Contains utility functions for web automation tasks like login, form filling, etc.
"""

import asyncio
import time
from typing import Dict, Any, Optional
from loguru import logger

try:
    from playwright.async_api import Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available for web tools")


class WebTools:
    """Web automation tools for common tasks"""
    
    @staticmethod
    async def login_suitecrm(page: Page, username: str = "user", password: str = "bitnami") -> bool:
        """
        Login to SuiteCRM with provided credentials
        
        Args:
            page: Playwright page object
            username: Login username (default: "user")
            password: Login password (default: "bitnami")
            
        Returns:
            bool: True if login successful, False otherwise
        """
        if not PLAYWRIGHT_AVAILABLE:
            logger.error("Playwright not available for login")
            return False
            
        try:
            logger.info(f"🔐 Attempting to login to SuiteCRM with username: {username}")
            
            # Wait for page to load
            await page.wait_for_load_state('networkidle')
            
            # Check if we're already logged in by looking for logout link or dashboard elements
            try:
                # Look for elements that indicate we're already logged in
                dashboard_indicators = [
                    'text=Dashboard',
                    'text=Accounts',
                    'text=Contacts',
                    'text=Leads',
                    'text=Opportunities',
                    '[data-testid="dashboard"]',
                    '.dashboard',
                    '#dashboard'
                ]
                
                for indicator in dashboard_indicators:
                    try:
                        await page.wait_for_selector(indicator, timeout=2000)
                        logger.info("✅ Already logged in to SuiteCRM")
                        return True
                    except:
                        continue
                        
            except Exception as e:
                logger.debug(f"Not already logged in: {e}")
            
            # Look for login form elements
            login_selectors = [
                # Common login form selectors
                'input[name="username"]',
                'input[name="user_name"]',
                'input[name="email"]',
                'input[type="text"]',
                'input[placeholder*="username" i]',
                'input[placeholder*="user" i]',
                'input[placeholder*="email" i]',
                '#username',
                '#user_name',
                '#email'
            ]
            
            password_selectors = [
                # Common password form selectors
                'input[name="password"]',
                'input[name="passwd"]',
                'input[type="password"]',
                'input[placeholder*="password" i]',
                'input[placeholder*="pass" i]',
                '#password',
                '#passwd'
            ]
            
            submit_selectors = [
                # Common submit button selectors
                'input[type="submit"]',
                'button[type="submit"]',
                'button:has-text("Login")',
                'button:has-text("Sign In")',
                'button:has-text("Log In")',
                'input[value*="Login" i]',
                'input[value*="Sign In" i]',
                'input[value*="Log In" i]',
                '.login-button',
                '#login-button',
                '#submit'
            ]
            
            # Find username field
            username_field = None
            for selector in login_selectors:
                try:
                    username_field = await page.wait_for_selector(selector, timeout=2000)
                    if username_field:
                        logger.info(f"✅ Found username field with selector: {selector}")
                        break
                except:
                    continue
            
            if not username_field:
                logger.error("❌ Could not find username field")
                return False
            
            # Find password field
            password_field = None
            for selector in password_selectors:
                try:
                    password_field = await page.wait_for_selector(selector, timeout=2000)
                    if password_field:
                        logger.info(f"✅ Found password field with selector: {selector}")
                        break
                except:
                    continue
            
            if not password_field:
                logger.error("❌ Could not find password field")
                return False
            
            # Find submit button
            submit_button = None
            for selector in submit_selectors:
                try:
                    submit_button = await page.wait_for_selector(selector, timeout=2000)
                    if submit_button:
                        logger.info(f"✅ Found submit button with selector: {selector}")
                        break
                except:
                    continue
            
            if not submit_button:
                logger.error("❌ Could not find submit button")
                return False
            
            # Fill in credentials
            await username_field.fill(username)
            await password_field.fill(password)
            
            # Wait a moment for any validation
            await asyncio.sleep(1)
            
            # Click submit button
            await submit_button.click()
            
            # Wait for navigation or dashboard to load
            try:
                await page.wait_for_load_state('networkidle', timeout=10000)
                
                # Check if login was successful by looking for dashboard elements
                success_indicators = [
                    'text=Dashboard',
                    'text=Accounts',
                    'text=Contacts',
                    'text=Leads',
                    'text=Opportunities',
                    'text=Welcome',
                    'text=SuiteCRM',
                    'text=Home',
                    'text=Profile',
                    'text=Logout',
                    '[data-testid="dashboard"]',
                    '.dashboard',
                    '#dashboard',
                    '.navbar',
                    '.main-content',
                    '.sidebar'
                ]
                
                for indicator in success_indicators:
                    try:
                        await page.wait_for_selector(indicator, timeout=3000)
                        logger.info("✅ Successfully logged in to SuiteCRM")
                        return True
                    except:
                        continue
                
                # Check if we got an error message
                error_indicators = [
                    'text=Invalid',
                    'text=Error',
                    'text=Failed',
                    'text=Incorrect',
                    '.error',
                    '.alert-danger',
                    '.login-error'
                ]
                
                for indicator in error_indicators:
                    try:
                        error_element = await page.wait_for_selector(indicator, timeout=2000)
                        if error_element:
                            error_text = await error_element.text_content()
                            logger.error(f"❌ Login failed: {error_text}")
                            return False
                    except:
                        continue
                
                # If we can't determine success/failure, check for login form disappearance
                try:
                    # Wait a bit longer for any redirects
                    await asyncio.sleep(2)
                    
                    # Check if login form is still present
                    login_form_selectors = [
                        'input[name="username"]',
                        'input[name="password"]',
                        'button[type="submit"]'
                    ]
                    
                    form_still_present = False
                    for selector in login_form_selectors:
                        try:
                            element = await page.wait_for_selector(selector, timeout=1000)
                            if element:
                                form_still_present = True
                                break
                        except:
                            continue
                    
                    if not form_still_present:
                        logger.info("✅ Login appears successful (login form no longer present)")
                        return True
                    else:
                        logger.warning("⚠️ Login form still present, login may have failed")
                        return False
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not determine login status: {e}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Error during login process: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Login failed with error: {e}")
            return False
    
    @staticmethod
    async def wait_for_page_load(page: Page, timeout: int = 10000) -> bool:
        """
        Wait for page to fully load
        
        Args:
            page: Playwright page object
            timeout: Timeout in milliseconds
            
        Returns:
            bool: True if page loaded successfully
        """
        try:
            await page.wait_for_load_state('networkidle', timeout=timeout)
            return True
        except Exception as e:
            logger.warning(f"Page load timeout: {e}")
            return False
    
    @staticmethod
    async def take_screenshot(page: Page, path: str) -> bool:
        """
        Take a screenshot of the current page
        
        Args:
            page: Playwright page object
            path: Path to save screenshot
            
        Returns:
            bool: True if screenshot saved successfully
        """
        try:
            await page.screenshot(path=path, full_page=True)
            logger.info(f"📸 Screenshot saved to: {path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to take screenshot: {e}")
            return False
