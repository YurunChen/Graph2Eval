-- Email Business Data Insertion Script for SuiteCRM
-- This script inserts realistic email templates and email records for testing

USE bitnami_suitecrm;

-- Insert Email Templates with business data
INSERT INTO email_templates (
    id, date_entered, date_modified, modified_user_id, created_by, 
    published, name, description, subject, body, body_html, 
    deleted, assigned_user_id, text_only, type
) VALUES
-- Business Email Templates
(
    'email-temp-001', NOW(), NOW(), '1', '1',
    'yes', 'Welcome Email Template', 
    'Welcome email for new customers with business information',
    'Welcome to Acme Corporation - Your Account Details',
    'Dear {{contact_name}},

Welcome to Acme Corporation! We are excited to have you as our valued customer.

Your account details:
- Account Number: {{account_number}}
- Contact Person: {{contact_name}}
- Email: {{contact_email}}
- Phone: {{contact_phone}}

Our business hours are Monday-Friday, 9:00 AM - 6:00 PM EST.

Best regards,
Acme Corporation Team
Phone: +1-555-123-4567
Email: support@acme.com',
    '<h2>Welcome to Acme Corporation!</h2>
<p>Dear {{contact_name}},</p>
<p>Welcome to Acme Corporation! We are excited to have you as our valued customer.</p>
<h3>Your Account Details:</h3>
<ul>
<li><strong>Account Number:</strong> {{account_number}}</li>
<li><strong>Contact Person:</strong> {{contact_name}}</li>
<li><strong>Email:</strong> {{contact_email}}</li>
<li><strong>Phone:</strong> {{contact_phone}}</li>
</ul>
<p>Our business hours are Monday-Friday, 9:00 AM - 6:00 PM EST.</p>
<p>Best regards,<br>
Acme Corporation Team<br>
Phone: +1-555-123-4567<br>
Email: support@acme.com</p>',
    0, '1', 0, 'business'
),
(
    'email-temp-002', NOW(), NOW(), '1', '1',
    'yes', 'Project Update Template',
    'Project status update email with business metrics',
    'Project Update: {{project_name}} - Status: {{project_status}}',
    'Hello {{contact_name}},

This is an update on your project: {{project_name}}

Current Status: {{project_status}}
Progress: {{project_progress}}%
Budget Used: ${{budget_used}}
Timeline: {{timeline_info}}

Key Milestones:
{{milestone_details}}

Next Steps:
{{next_steps}}

If you have any questions, please contact:
{{project_manager}}
{{manager_phone}}
{{manager_email}}

Best regards,
Project Management Team',
    '<h2>Project Update: {{project_name}}</h2>
<p>Hello {{contact_name}},</p>
<p>This is an update on your project: <strong>{{project_name}}</strong></p>
<h3>Current Status:</h3>
<ul>
<li><strong>Status:</strong> {{project_status}}</li>
<li><strong>Progress:</strong> {{project_progress}}%</li>
<li><strong>Budget Used:</strong> ${{budget_used}}</li>
<li><strong>Timeline:</strong> {{timeline_info}}</li>
</ul>
<h3>Key Milestones:</h3>
<p>{{milestone_details}}</p>
<h3>Next Steps:</h3>
<p>{{next_steps}}</p>
<p>If you have any questions, please contact:<br>
{{project_manager}}<br>
{{manager_phone}}<br>
{{manager_email}}</p>
<p>Best regards,<br>
Project Management Team</p>',
    0, '1', 0, 'business'
),
(
    'email-temp-003', NOW(), NOW(), '1', '1',
    'yes', 'Invoice Template',
    'Professional invoice email with business details',
    'Invoice #{{invoice_number}} - {{company_name}} - Amount: ${{invoice_amount}}',
    'Dear {{contact_name}},

Please find attached invoice #{{invoice_number}} for services rendered.

Invoice Details:
- Invoice Number: {{invoice_number}}
- Date: {{invoice_date}}
- Due Date: {{due_date}}
- Amount: ${{invoice_amount}}
- Tax: ${{tax_amount}}
- Total: ${{total_amount}}

Services Provided:
{{service_details}}

Payment Terms: Net 30 days
Payment Methods: Bank Transfer, Credit Card, Check

For payment questions, contact:
{{accounting_contact}}
{{accounting_phone}}
{{accounting_email}}

Thank you for your business!

Best regards,
{{company_name}} Accounting Team',
    '<h2>Invoice #{{invoice_number}}</h2>
<p>Dear {{contact_name}},</p>
<p>Please find attached invoice #{{invoice_number}} for services rendered.</p>
<h3>Invoice Details:</h3>
<ul>
<li><strong>Invoice Number:</strong> {{invoice_number}}</li>
<li><strong>Date:</strong> {{invoice_date}}</li>
<li><strong>Due Date:</strong> {{due_date}}</li>
<li><strong>Amount:</strong> ${{invoice_amount}}</li>
<li><strong>Tax:</strong> ${{tax_amount}}</li>
<li><strong>Total:</strong> ${{total_amount}}</li>
</ul>
<h3>Services Provided:</h3>
<p>{{service_details}}</p>
<p><strong>Payment Terms:</strong> Net 30 days<br>
<strong>Payment Methods:</strong> Bank Transfer, Credit Card, Check</p>
<p>For payment questions, contact:<br>
{{accounting_contact}}<br>
{{accounting_phone}}<br>
{{accounting_email}}</p>
<p>Thank you for your business!</p>
<p>Best regards,<br>
{{company_name}} Accounting Team</p>',
    0, '1', 0, 'business'
),
(
    'email-temp-004', NOW(), NOW(), '1', '1',
    'yes', 'Meeting Invitation Template',
    'Business meeting invitation with agenda',
    'Meeting Invitation: {{meeting_topic}} - {{meeting_date}} at {{meeting_time}}',
    'Hello {{attendee_name}},

You are invited to attend the following business meeting:

Meeting Topic: {{meeting_topic}}
Date: {{meeting_date}}
Time: {{meeting_time}}
Location: {{meeting_location}}
Duration: {{meeting_duration}}

Agenda:
{{meeting_agenda}}

Participants:
{{participant_list}}

Please confirm your attendance by replying to this email or calling {{organizer_phone}}.

If you cannot attend, please let us know at least 24 hours in advance.

Best regards,
{{organizer_name}}
{{organizer_title}}
{{organizer_phone}}
{{organizer_email}}',
    '<h2>Meeting Invitation: {{meeting_topic}}</h2>
<p>Hello {{attendee_name}},</p>
<p>You are invited to attend the following business meeting:</p>
<h3>Meeting Details:</h3>
<ul>
<li><strong>Topic:</strong> {{meeting_topic}}</li>
<li><strong>Date:</strong> {{meeting_date}}</li>
<li><strong>Time:</strong> {{meeting_time}}</li>
<li><strong>Location:</strong> {{meeting_location}}</li>
<li><strong>Duration:</strong> {{meeting_duration}}</li>
</ul>
<h3>Agenda:</h3>
<p>{{meeting_agenda}}</p>
<h3>Participants:</h3>
<p>{{participant_list}}</p>
<p>Please confirm your attendance by replying to this email or calling {{organizer_phone}}.</p>
<p>If you cannot attend, please let us know at least 24 hours in advance.</p>
<p>Best regards,<br>
{{organizer_name}}<br>
{{organizer_title}}<br>
{{organizer_phone}}<br>
{{organizer_email}}</p>',
    0, '1', 0, 'business'
),
(
    'email-temp-005', NOW(), NOW(), '1', '1',
    'yes', 'Lead Follow-up Template',
    'Lead follow-up email with business information',
    'Follow-up: {{lead_name}} - {{company_name}} - {{lead_source}}',
    'Dear {{lead_name}},

Thank you for your interest in {{company_name}}. We appreciate you taking the time to learn about our services.

Lead Information:
- Lead Name: {{lead_name}}
- Company: {{company_name}}
- Position: {{lead_position}}
- Phone: {{lead_phone}}
- Email: {{lead_email}}
- Source: {{lead_source}}

Our Services:
{{service_offerings}}

Next Steps:
{{next_actions}}

We would like to schedule a brief call to discuss how we can help {{company_name}} achieve its goals.

Please let us know your preferred time, or call us at {{sales_phone}}.

Best regards,
{{sales_rep_name}}
{{sales_rep_title}}
{{sales_phone}}
{{sales_email}}',
    '<h2>Follow-up: {{lead_name}}</h2>
<p>Dear {{lead_name}},</p>
<p>Thank you for your interest in {{company_name}}. We appreciate you taking the time to learn about our services.</p>
<h3>Lead Information:</h3>
<ul>
<li><strong>Lead Name:</strong> {{lead_name}}</li>
<li><strong>Company:</strong> {{company_name}}</li>
<li><strong>Position:</strong> {{lead_position}}</li>
<li><strong>Phone:</strong> {{lead_phone}}</li>
<li><strong>Email:</strong> {{lead_email}}</li>
<li><strong>Source:</strong> {{lead_source}}</li>
</ul>
<h3>Our Services:</h3>
<p>{{service_offerings}}</p>
<h3>Next Steps:</h3>
<p>{{next_actions}}</p>
<p>We would like to schedule a brief call to discuss how we can help {{company_name}} achieve its goals.</p>
<p>Please let us know your preferred time, or call us at {{sales_phone}}.</p>
<p>Best regards,<br>
{{sales_rep_name}}<br>
{{sales_rep_title}}<br>
{{sales_phone}}<br>
{{sales_email}}</p>',
    0, '1', 0, 'business'
);

-- Insert Email Records
INSERT INTO emails (
    id, name, date_entered, date_modified, modified_user_id, created_by,
    deleted, assigned_user_id, orphaned, date_sent_received, message_id,
    type, status, flagged, reply_to_status, intent, mailbox_id,
    parent_type, parent_id, uid, category_id
) VALUES
(
    'email-001', 'Welcome Email - John Smith', NOW(), NOW(), '1', '1',
    0, '1', 0, NOW(), 'welcome-001@acme.com',
    'out', 'sent', 0, 0, 'send', NULL,
    'Contacts', 'contact-001', 'welcome-001', 'welcome'
),
(
    'email-002', 'Project Update - Website Redesign', NOW(), NOW(), '1', '1',
    0, '1', 0, NOW(), 'project-update-001@acme.com',
    'out', 'sent', 0, 0, 'send', NULL,
    'Accounts', 'account-001', 'project-001', 'project'
),
(
    'email-003', 'Invoice #INV-2024-001 - Acme Corp', NOW(), NOW(), '1', '1',
    0, '1', 0, NOW(), 'invoice-001@acme.com',
    'out', 'sent', 0, 0, 'send', NULL,
    'Accounts', 'account-002', 'invoice-001', 'invoice'
),
(
    'email-004', 'Meeting Invitation - Q4 Review', NOW(), NOW(), '1', '1',
    0, '1', 0, NOW(), 'meeting-001@acme.com',
    'out', 'sent', 0, 0, 'send', NULL,
    'Contacts', 'contact-002', 'meeting-001', 'meeting'
),
(
    'email-005', 'Lead Follow-up - Sarah Johnson', NOW(), NOW(), '1', '1',
    0, '1', 0, NOW(), 'lead-followup-001@acme.com',
    'out', 'sent', 0, 0, 'send', NULL,
    'Leads', 'lead-001', 'lead-001', 'lead'
);

-- Insert Email Text Content
INSERT INTO emails_text (
    email_id, from_addr, reply_to_addr, to_addrs, cc_addrs, bcc_addrs,
    description, description_html, raw_source, deleted
) VALUES
(
    'email-001', 'support@acme.com', 'support@acme.com',
    'john.smith@globex.com', 'sales@acme.com', '',
    'Dear John Smith,

Welcome to Acme Corporation! We are excited to have you as our valued customer.

Your account details:
- Account Number: ACC-2024-001
- Contact Person: John Smith
- Email: john.smith@globex.com
- Phone: +1-555-123-4567

Our business hours are Monday-Friday, 9:00 AM - 6:00 PM EST.

Best regards,
Acme Corporation Team
Phone: +1-555-123-4567
Email: support@acme.com',
    '<h2>Welcome to Acme Corporation!</h2>
<p>Dear John Smith,</p>
<p>Welcome to Acme Corporation! We are excited to have you as our valued customer.</p>
<h3>Your Account Details:</h3>
<ul>
<li><strong>Account Number:</strong> ACC-2024-001</li>
<li><strong>Contact Person:</strong> John Smith</li>
<li><strong>Email:</strong> john.smith@globex.com</li>
<li><strong>Phone:</strong> +1-555-123-4567</li>
</ul>
<p>Our business hours are Monday-Friday, 9:00 AM - 6:00 PM EST.</p>
<p>Best regards,<br>
Acme Corporation Team<br>
Phone: +1-555-123-4567<br>
Email: support@acme.com</p>',
    'Raw email source content here...', 0
),
(
    'email-002', 'project@acme.com', 'project@acme.com',
    'sarah.johnson@soylent.com', 'pm@acme.com', '',
    'Hello Sarah Johnson,

This is an update on your project: Website Redesign

Current Status: In Progress
Progress: 75%
Budget Used: $45,000
Timeline: On track for March 15th completion

Key Milestones:
- Design phase completed
- Development phase 75% complete
- Testing phase starting next week

Next Steps:
- Complete remaining development tasks
- Begin user acceptance testing
- Prepare for deployment

If you have any questions, please contact:
Mike Chen, Project Manager
+1-555-987-6543
mike.chen@acme.com

Best regards,
Project Management Team',
    '<h2>Project Update: Website Redesign</h2>
<p>Hello Sarah Johnson,</p>
<p>This is an update on your project: <strong>Website Redesign</strong></p>
<h3>Current Status:</h3>
<ul>
<li><strong>Status:</strong> In Progress</li>
<li><strong>Progress:</strong> 75%</li>
<li><strong>Budget Used:</strong> $45,000</li>
<li><strong>Timeline:</strong> On track for March 15th completion</li>
</ul>
<h3>Key Milestones:</h3>
<p>- Design phase completed<br>
- Development phase 75% complete<br>
- Testing phase starting next week</p>
<h3>Next Steps:</h3>
<p>- Complete remaining development tasks<br>
- Begin user acceptance testing<br>
- Prepare for deployment</p>
<p>If you have any questions, please contact:<br>
Mike Chen, Project Manager<br>
+1-555-987-6543<br>
mike.chen@acme.com</p>
<p>Best regards,<br>
Project Management Team</p>',
    'Raw email source content here...', 0
),
(
    'email-003', 'accounting@acme.com', 'accounting@acme.com',
    'billing@initech.com', 'finance@acme.com', '',
    'Dear David Wilson,

Please find attached invoice #INV-2024-001 for services rendered.

Invoice Details:
- Invoice Number: INV-2024-001
- Date: 2024-01-15
- Due Date: 2024-02-14
- Amount: $12,500.00
- Tax: $1,250.00
- Total: $13,750.00

Services Provided:
- Website Development Services
- Database Integration
- User Training Sessions

Payment Terms: Net 30 days
Payment Methods: Bank Transfer, Credit Card, Check

For payment questions, contact:
Lisa Rodriguez, Accounting Manager
+1-555-456-7890
lisa.rodriguez@acme.com

Thank you for your business!

Best regards,
Acme Corporation Accounting Team',
    '<h2>Invoice #INV-2024-001</h2>
<p>Dear David Wilson,</p>
<p>Please find attached invoice #INV-2024-001 for services rendered.</p>
<h3>Invoice Details:</h3>
<ul>
<li><strong>Invoice Number:</strong> INV-2024-001</li>
<li><strong>Date:</strong> 2024-01-15</li>
<li><strong>Due Date:</strong> 2024-02-14</li>
<li><strong>Amount:</strong> $12,500.00</li>
<li><strong>Tax:</strong> $1,250.00</li>
<li><strong>Total:</strong> $13,750.00</li>
</ul>
<h3>Services Provided:</h3>
<p>- Website Development Services<br>
- Database Integration<br>
- User Training Sessions</p>
<p><strong>Payment Terms:</strong> Net 30 days<br>
<strong>Payment Methods:</strong> Bank Transfer, Credit Card, Check</p>
<p>For payment questions, contact:<br>
Lisa Rodriguez, Accounting Manager<br>
+1-555-456-7890<br>
lisa.rodriguez@acme.com</p>
<p>Thank you for your business!</p>
<p>Best regards,<br>
Acme Corporation Accounting Team</p>',
    'Raw email source content here...', 0
),
(
    'email-004', 'meetings@acme.com', 'meetings@acme.com',
    'emily.brown@umbrella.com', 'executive@acme.com', '',
    'Hello Emily Brown,

You are invited to attend the following business meeting:

Meeting Topic: Q4 Business Review
Date: 2024-01-20
Time: 2:00 PM EST
Location: Conference Room A, Acme Headquarters
Duration: 90 minutes

Agenda:
- Q4 Financial Results Review
- Annual Performance Metrics
- Strategic Planning for Q1 2024
- Budget Allocation Discussion
- Team Performance Recognition

Participants:
- Emily Brown (Umbrella Corporation)
- John Smith (Acme Corporation)
- Sarah Johnson (Soylent Corp)
- David Wilson (Initech)

Please confirm your attendance by replying to this email or calling +1-555-789-0123.

If you cannot attend, please let us know at least 24 hours in advance.

Best regards,
Jennifer Davis
Executive Assistant
+1-555-789-0123
jennifer.davis@acme.com',
    '<h2>Meeting Invitation: Q4 Business Review</h2>
<p>Hello Emily Brown,</p>
<p>You are invited to attend the following business meeting:</p>
<h3>Meeting Details:</h3>
<ul>
<li><strong>Topic:</strong> Q4 Business Review</li>
<li><strong>Date:</strong> 2024-01-20</li>
<li><strong>Time:</strong> 2:00 PM EST</li>
<li><strong>Location:</strong> Conference Room A, Acme Headquarters</li>
<li><strong>Duration:</strong> 90 minutes</li>
</ul>
<h3>Agenda:</h3>
<p>- Q4 Financial Results Review<br>
- Annual Performance Metrics<br>
- Strategic Planning for Q1 2024<br>
- Budget Allocation Discussion<br>
- Team Performance Recognition</p>
<h3>Participants:</h3>
<p>- Emily Brown (Umbrella Corporation)<br>
- John Smith (Acme Corporation)<br>
- Sarah Johnson (Soylent Corp)<br>
- David Wilson (Initech)</p>
<p>Please confirm your attendance by replying to this email or calling +1-555-789-0123.</p>
<p>If you cannot attend, please let us know at least 24 hours in advance.</p>
<p>Best regards,<br>
Jennifer Davis<br>
Executive Assistant<br>
+1-555-789-0123<br>
jennifer.davis@acme.com</p>',
    'Raw email source content here...', 0
),
(
    'email-005', 'sales@acme.com', 'sales@acme.com',
    'robert.garcia@massive.com', 'leads@acme.com', '',
    'Dear Robert Garcia,

Thank you for your interest in Acme Corporation. We appreciate you taking the time to learn about our services.

Lead Information:
- Lead Name: Robert Garcia
- Company: Massive Dynamic
- Position: IT Director
- Phone: +1-555-321-6547
- Email: robert.garcia@massive.com
- Source: Website Contact Form

Our Services:
- Custom Software Development
- Cloud Migration Services
- Cybersecurity Solutions
- Data Analytics Implementation
- IT Consulting

Next Steps:
- Schedule initial consultation call
- Discuss specific project requirements
- Provide detailed proposal
- Arrange technical demonstration

We would like to schedule a brief call to discuss how we can help Massive Dynamic achieve its goals.

Please let us know your preferred time, or call us at +1-555-111-2222.

Best regards,
Alex Thompson
Senior Sales Representative
+1-555-111-2222
alex.thompson@acme.com',
    '<h2>Follow-up: Robert Garcia</h2>
<p>Dear Robert Garcia,</p>
<p>Thank you for your interest in Acme Corporation. We appreciate you taking the time to learn about our services.</p>
<h3>Lead Information:</h3>
<ul>
<li><strong>Lead Name:</strong> Robert Garcia</li>
<li><strong>Company:</strong> Massive Dynamic</li>
<li><strong>Position:</strong> IT Director</li>
<li><strong>Phone:</strong> +1-555-321-6547</li>
<li><strong>Email:</strong> robert.garcia@massive.com</li>
<li><strong>Source:</strong> Website Contact Form</li>
</ul>
<h3>Our Services:</h3>
<p>- Custom Software Development<br>
- Cloud Migration Services<br>
- Cybersecurity Solutions<br>
- Data Analytics Implementation<br>
- IT Consulting</p>
<h3>Next Steps:</h3>
<p>- Schedule initial consultation call<br>
- Discuss specific project requirements<br>
- Provide detailed proposal<br>
- Arrange technical demonstration</p>
<p>We would like to schedule a brief call to discuss how we can help Massive Dynamic achieve its goals.</p>
<p>Please let us know your preferred time, or call us at +1-555-111-2222.</p>
<p>Best regards,<br>
Alex Thompson<br>
Senior Sales Representative<br>
+1-555-111-2222<br>
alex.thompson@acme.com</p>',
    'Raw email source content here...', 0
);

-- Display inserted data
SELECT 'Email Templates inserted:' as info;
SELECT id, name, subject, type FROM email_templates WHERE id LIKE 'email-temp-%';

SELECT 'Email Records inserted:' as info;
SELECT id, name, type, status FROM emails WHERE id LIKE 'email-%';

SELECT 'Email Text Content inserted:' as info;
SELECT email_id, from_addr, to_addrs FROM emails_text WHERE email_id LIKE 'email-%';

