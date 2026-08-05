# Placeholder for vendored gigatoken

Local/CI builds that omit a real checkout leave this directory without
`Cargo.toml`, so the Dockerfile falls back to cloning
`GIGATOKEN_REPO` @ `GIGATOKEN_COMMIT`.

On the Plan A EC2 host, `artifacts/plan_a_ec2_userdata.sh` extracts
`supergigatoken-00e61db.tar.gz` over this path (including `.git`) before
`docker build`, so the image installs the unpublished pin without a
GitHub push.
