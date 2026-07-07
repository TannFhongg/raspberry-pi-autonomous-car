# Security

## Reporting

If you find a security or safety-sensitive issue, please report it privately to the repository owner instead of opening a public issue first.

## Scope of Security Here

For this project, security includes both software and hardware-adjacent concerns:

- network-exposed Flask dashboards
- unsafe serial command behavior
- unexpected motor activation
- missing safeguards around runtime failures

## Current Practical Risks

- the dashboards do not include authentication
- multiple tools expose HTTP services on the local network
- robot control is safety-sensitive even in a hobby or portfolio setting

## Recommended Handling

- do not expose the robot control dashboards to untrusted networks
- validate serial and motor changes carefully before field tests
- keep a physical stop method available during testing
