import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime

class EmailManager:
    """
    Email Manager for Trading Workflow
    Handles initial orders and adjustment emails with Gmail SMTP
    """
    
    def __init__(self):
        # Fixed email configuration
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = "lucas.afonso@hashdex.com"
        self.app_password = "scpb vuvs lfiu hcml"
        self.recipient_email = "ets@xpi.com.br"
    
    def _send_gmail_with_confirmation(self, subject, body, 
                                      cc_emails=['trading@hashdex.com', 'operations@hashdex.com'], attachment_path=None):
        """
        Send email through Gmail SMTP with user confirmation
        
        Args:
            subject (str): Email subject
            body (str): Email body (HTML)
            cc_emails (list, optional): List of CC email addresses
            attachment_path (str, optional): Path to file attachment
        """
        # Show email preview
        print("=" * 60)
        print("📧 EMAIL PREVIEW")
        print("=" * 60)
        print(f"From: {self.sender_email}")
        print(f"To: {self.recipient_email}")
        if cc_emails:
            print(f"CC: {', '.join(cc_emails)}")
        print(f"Subject: {subject}")
        print("-" * 40)
        print("Body (HTML):")
        print(body)
        print("=" * 60)
        
        # Ask for confirmation
        while True:
            confirm = input("\n🤔 Do you want to send this email? (y/n): ").lower().strip()
            if confirm in ['y', 'yes', 's', 'sim']:
                break
            elif confirm in ['n', 'no', 'nao', 'não']:
                print("❌ Email cancelled by user")
                return False
            else:
                print("Please answer 'y' for yes or 'n' for no")
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = subject
            
            # Add CC if provided
            if cc_emails:
                msg['Cc'] = ', '.join(cc_emails)
                all_recipients = [self.recipient_email] + cc_emails
            else:
                all_recipients = [self.recipient_email]
            
            # Add body as HTML
            msg.attach(MIMEText(body, 'html'))
            
            # Add attachment if provided
            if attachment_path and os.path.exists(attachment_path):
                with open(attachment_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {os.path.basename(attachment_path)}'
                    )
                    msg.attach(part)
            
            # Send email
            print("📤 Sending email...")
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.app_password)
            server.send_message(msg, to_addrs=all_recipients)
            server.quit()
            
            print(f"✅ Email sent successfully to {self.recipient_email}")
            if cc_emails:
                print(f"📋 CC sent to: {', '.join(cc_emails)}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send email: {str(e)}")
            return False
    
    def _convert_text_to_html_table(self, text_section):
        """
        Convert tab-separated table text to HTML table
        """
        lines = text_section.strip().split('\n')
        if len(lines) < 2:
            return text_section
        
        # First line is header
        header = lines[0].split('\t')
        rows = [line.split('\t') for line in lines[1:] if line.strip()]
        
        html = '<table border="1" style="border-collapse: collapse; margin: 10px 0; width: 100%;">\n'
        
        # Header
        html += '  <thead style="background-color: #f0f0f0;">\n'
        html += '    <tr>\n'
        for col in header:
            html += f'      <th style="padding: 8px; text-align: left;">{col.strip()}</th>\n'
        html += '    </tr>\n'
        html += '  </thead>\n'
        
        # Body
        html += '  <tbody>\n'
        for row in rows:
            html += '    <tr>\n'
            for col in row:
                html += f'      <td style="padding: 8px;">{col.strip()}</td>\n'
            html += '    </tr>\n'
        html += '  </tbody>\n'
        html += '</table>\n'
        
        return html
    
    def _parse_adjustment_email_body(self, email_body):
        """
        Parse the adjustment email body and convert tables to HTML
        Reorganizes content: intro -> ORDENS ESPECÍFICAS -> observations -> tables
        """
        lines = email_body.split('\n')
        
        # Storage for different sections
        intro_parts = []
        specific_orders = []
        observations = []
        table_parts = []
        
        current_section = []
        in_table = False
        in_specific_orders = False
        
        for line in lines:
            # Check if we're starting the specific orders section
            if '=== ORDENS ESPECÍFICAS ===' in line:
                # Save any current section to intro (but skip observations)
                if current_section and not any(line.startswith(('Obs:', 'Obs2:')) for line in current_section):
                    intro_parts.append('<p>' + '<br>'.join(current_section) + '</p>')
                current_section = []
                in_specific_orders = True
                specific_orders.append('<h3>🎯 ' + line.replace('===', '').strip() + '</h3>')
                continue
            
            # Check if this line is an observation
            elif line.strip().startswith(('Obs:', 'Obs2:')):
                # Save current section appropriately
                if current_section:
                    if in_specific_orders:
                        specific_orders.append('<p>' + '<br>'.join(current_section) + '</p>')
                    else:
                        intro_parts.append('<p>' + '<br>'.join(current_section) + '</p>')
                    current_section = []
                
                # Start collecting observations
                observations.append(line.strip())
                in_specific_orders = False
                continue
            
            # Check if this is a table header line
            elif 'ETF <compra>' in line and '\t' in line:
                # Save any remaining content
                if current_section:
                    if in_specific_orders:
                        specific_orders.append('<p>' + '<br>'.join(current_section) + '</p>')
                    else:
                        intro_parts.append('<p>' + '<br>'.join(current_section) + '</p>')
                    current_section = []
                
                in_specific_orders = False
                in_table = True
                table_lines = [line]
                continue
            
            elif 'ETF <venda>' in line and '\t' in line:
                # Save any remaining content
                if current_section:
                    if in_specific_orders:
                        specific_orders.append('<p>' + '<br>'.join(current_section) + '</p>')
                    else:
                        intro_parts.append('<p>' + '<br>'.join(current_section) + '</p>')
                    current_section = []
                
                in_specific_orders = False
                in_table = True
                table_lines = [line]
                continue
            
            elif in_table and '\t' in line and line.strip():
                table_lines.append(line)
                continue
            
            elif in_table and (not line.strip() or '\t' not in line):
                # End of table - convert to HTML and add to table_parts
                table_text = '\n'.join(table_lines)
                html_table = self._convert_text_to_html_table(table_text)
                table_parts.append(html_table)
                in_table = False
                if line.strip():
                    current_section.append(line)
                continue
            
            else:
                # Regular text
                if line.strip():
                    current_section.append(line)
                elif current_section:
                    if in_specific_orders:
                        specific_orders.append('<p>' + '<br>'.join(current_section) + '</p>')
                    else:
                        intro_parts.append('<p>' + '<br>'.join(current_section) + '</p>')
                    current_section = []
        
        # Handle any remaining content
        if in_table and 'table_lines' in locals():
            table_text = '\n'.join(table_lines)
            html_table = self._convert_text_to_html_table(table_text)
            table_parts.append(html_table)
        elif current_section:
            if in_specific_orders:
                specific_orders.append('<p>' + '<br>'.join(current_section) + '</p>')
            else:
                intro_parts.append('<p>' + '<br>'.join(current_section) + '</p>')
        
        # Add Obs3 about TWAP execution
        observations.append('Obs3: Favor executar um TWAP até 15:30')
        
        # Convert observations to HTML
        observations_html = []
        if observations:
            observations_html.append('<h3>📋 Observações:</h3>')
            for obs in observations:
                observations_html.append(f'<p><strong>{obs}</strong></p>')
        
        # Combine in the desired order: intro -> specific orders -> observations -> tables
        final_html = []
        final_html.extend(intro_parts)
        final_html.extend(specific_orders)
        final_html.extend(observations_html)
        final_html.extend(table_parts)
        
        return '\n'.join(final_html)
    
    def send_initial_orders_email(self, initial_function, cc_emails=['trading@hashdex.com', 'operations@hashdex.com']):
        """
        📧 Send initial orders email with confirmation
        
        Args:
            initial_function: Result from workflow_manager.run_initial_orders()
            cc_emails (list, optional): List of CC email addresses
        """
        # Get the sentences
        sentences = initial_function['initial_orders']["sentences"]
        
        # Get current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Prepare email content
        subject = f"Hashdex <> XP B3 Orders - {current_date}"
        
        # Create HTML email body
        body = f"""
        <h2>📈 Initial Orders - Morning Routine</h2>
        <p><strong>Date:</strong> {current_date}</p>
        <p><strong>Total Orders:</strong> {len(sentences)}</p>
        
        <hr>
        
        <h3>🎯 ORDENS ESPECÍFICAS</h3>
        """
        
        # Add each sentence as a specific order
        for i, sentence in enumerate(sentences, 1):
            body += f"<p>{i}. {sentence}</p>\n"
        
        body += """
        
        <h3>📋 Observações:</h3>
        <p><strong>Obs: Favor executar as ordens como um TWAP até as 12:40</strong></p>

        <br>
        <p>---------------------------------------------------</p>
        <p><em>Investment Management Team</em><br>
        Hashdex.com.br</p>
        <p>---------------------------------------------------</p>

        <hr>
        <p><em>Generated automatically by Trading Workflow Manager</em></p>
        """
        
        # Ask if user wants to send email
        print(f"\n📧 Prepare to send initial orders email to {self.recipient_email}")
        
        # Send email with confirmation
        email_sent = self._send_gmail_with_confirmation(
            subject=subject,
            body=body,
            cc_emails=cc_emails
        )
        
        if email_sent:
            print("✅ Initial orders workflow completed with email notification")
            return True
        else:
            print("ℹ️ Initial orders workflow completed (no email sent)")
            return False
    
    def send_adjustment_email(self, adjustment_results, iteration=0, cc_emails=['trading@hashdex.com', 'operations@hashdex.com']):
        """
        📊 Send adjustment cycle email with proper table formatting
        
        Args:
            adjustment_results: The result from workflow_manager.run_adjustment_cycle()
            iteration: Which iteration to send (default: 0 for first/latest)
            cc_emails (list, optional): List of CC email addresses
        """
        # Extract email body from adjustment results
        email_body = adjustment_results['adjustment_results'][iteration]['email_body']
        
        # Get current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Prepare email content
        subject = f"Hashdex <> XP B3 Adjustment - {current_date}"
        
        # Parse and convert tables to HTML
        parsed_body = self._parse_adjustment_email_body(email_body)
        
        # Create enhanced HTML email body
        body = f"""
        <h2>📊 Trading Adjustment Instructions</h2>
        <p><strong>Date:</strong> {current_date}</p>
        <p><strong>Iteration:</strong> {iteration + 1}</p>
        
        <hr>
        
        {parsed_body}
        
        <br><br>
        <p>-------------------------------</p>
        <p><em>Investment Management Team</em><br>
        Hashdex.com.br</p>
        <p>--------------------------------</p>
        
        <hr>
        <p><em>Generated automatically by Trading Workflow Manager</em></p>
        """
        
        # Ask if user wants to send email
        print(f"\n📧 Prepare to send adjustment email to {self.recipient_email}")
        
        # Send email with confirmation
        email_sent = self._send_gmail_with_confirmation(
            subject=subject,
            body=body,
            cc_emails=cc_emails
        )
        
        if email_sent:
            print("✅ Adjustment email sent successfully")
            return True
        else:
            print("ℹ️ Adjustment email cancelled")
            return False


# Backward compatibility functions (optional)
def send_initial_orders_email(initial_function, 
                              cc_emails=['trading@hashdex.com', 'operations@hashdex.com']):
    """Backward compatibility wrapper"""
    email_manager = EmailManager()
    return email_manager.send_initial_orders_email(initial_function, cc_emails)

def send_adjustment_email(adjustment_results, iteration=0, 
                          cc_emails=['trading@hashdex.com', 'operations@hashdex.com']):
    """Backward compatibility wrapper"""
    email_manager = EmailManager()
    return email_manager.send_adjustment_email(adjustment_results, iteration, cc_emails)

